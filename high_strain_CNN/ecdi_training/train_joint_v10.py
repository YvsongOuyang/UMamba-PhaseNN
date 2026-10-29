"""Bounded joint-cascade pilot on the full immutable training split.

Randomly visits training samples without replacement. Scores on ALL validation
samples, never test. BN statistics and dropout stay in inference mode, as in
the V9 fitting diagnostic. Source checkpoint is never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aligned_objective_v7 import OBJECTIVE_VERSION, objective_and_metrics
from .data_v2 import build_dataset, prepare_batch
from .train import load_checkpoint, make_model, verify_data_identity
from .train_aligned_v7 import evaluate, file_digest, read_thickness, save_results
from .train_phase_cascade_v8 import phase_bn_digest

LOG = logging.getLogger(__name__)
VERSION = 'joint_cascade_v10_pilot'
PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='/data_ssd/oyys/eCDI')
    parser.add_argument('--run-name', default='joint_cascade_v10_pilot')
    parser.add_argument('--checkpoint-root', type=Path,
                        default=Path('/data_ssd/oyys/high_strain_cnn/pipeline_output_2d'))
    parser.add_argument('--runs-dir', type=Path, default=PROJECT / 'artifacts/training/ecdi_training')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation-steps', type=int, default=3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--max-steps', type=int, default=240)
    parser.add_argument('--check-every', type=int, default=120)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    args = parser.parse_args()
    if (min(args.batch_size, args.accumulation_steps, args.max_steps, args.check_every) < 1
            or args.max_steps > 240 or args.num_workers < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('Positive batches/steps/LR required; max-steps <= 240; workers >= 0.')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('Run name must be a single directory name.')
    root = args.runs_dir / args.run_name
    ckpt_dir = args.checkpoint_root / args.run_name
    if root.exists() or ckpt_dir.exists():
        raise FileExistsError('Use a fresh run name. Existing outputs are preserved.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    source = load_checkpoint(args.checkpoint)
    if source.get('stage') != 'refiner' or source.get('experiment_version') != 'phase_through_fixed_refiner_v8':
        raise ValueError('Start from the selected complete V8 cascade.')
    if source.get('registered_protocol', {}).get('search_radius') != 4.:
        raise ValueError('Source must use the V7/V8 registration protocol, radius=4.')
    config = copy.deepcopy(source['config'])
    config['root'] = str(Path(args.data_dir).resolve())
    datasets = {}
    try:
        for split in ('train', 'val'):
            datasets[split] = build_dataset(config, split, include_ideal=split == 'val')
            actual = datasets[split].manifest
            verify_data_identity(source, actual)
            previous = source.get('data_manifests', {}).get(split, {})
            for key in ('config_sha256', 'manifest_sha256', 'selected_indices_sha256'):
                if not previous.get(key) or previous[key] != actual.get(key):
                    raise ValueError(f'Source/current {split} identity differs: {key}')
            if actual['selected_samples'] != actual['available_samples']:
                raise ValueError('Datasets must use the complete original split.')
        generator = torch.Generator().manual_seed(42)
        train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, generator=generator, drop_last=False)
        val_loader = DataLoader(datasets['val'], batch_size=12, shuffle=False,
                                num_workers=args.num_workers, drop_last=False)
        thickness = read_thickness(datasets['val'])
        model = make_model('refiner', config, checkpoint=source).to(device)
        model.freeze_phase_model = False
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        # Gradients remain enabled in eval mode; dropout/BN running updates do not.
        model.eval()
        bn_hash = phase_bn_digest(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        root.mkdir(parents=True)
        logging.basicConfig(level=logging.INFO, force=True,
                            format='%(asctime)s | %(levelname)s | %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler(root / 'console.log')])
        lineage = {'source_checkpoint': str(Path(args.checkpoint).resolve()),
                   'source_sha256': file_digest(args.checkpoint), 'source_epoch': source['epoch']}
        protocol = {**vars(args), 'objective': OBJECTIVE_VERSION, 'search_radius': 4.,
                    'dropout': 'off', 'bn_running_stats': 'frozen',
                    'accept_min_relative_val_gain': .02,
                    'accept_max_material_relative_regression': .02}
        # Path objects are converted only for JSON; actual checkpoint config is native.
        (root / 'config.json').write_text(json.dumps({'protocol': protocol, 'lineage': lineage},
                                                    indent=2, default=str), encoding='utf-8')
        reference, rows = evaluate(model, val_loader, config, device, thickness, 4.)
        save_results(root, 'reference', reference, rows)
        reference_score = reference['refined']['all']['registered_complex_nrmse']
        best_score = reference_score
        selected_checkpoint = str(Path(args.checkpoint).resolve())
        history = []
        started = time.monotonic()
        LOG.info('Joint phase + refiner | full TRAIN pool=%d | full VAL=%d | effective_batch=%d',
                 len(datasets['train']), len(datasets['val']), args.batch_size * args.accumulation_steps)
        LOG.info('Reference val=%.6f | max_updates=%d | check_every=%d',
                 reference_score, args.max_steps, args.check_every)
        optimizer.zero_grad(set_to_none=True)
        seen = group_samples = micro_count = steps = 0
        effective_batch = args.batch_size * args.accumulation_steps
        stop_reason = 'update_limit'
        config['experiment_version'] = VERSION

        def validate():
            nonlocal best_score, selected_checkpoint
            if phase_bn_digest(model) != bn_hash:
                raise RuntimeError('Frozen phase BatchNorm statistics changed.')
            metrics, records = evaluate(model, val_loader, config, device, thickness, 4.)
            current = metrics['refined']['all']['registered_complex_nrmse']
            if not math.isfinite(current):
                raise ValueError('Nonfinite validation score.')
            gain = 1. - current / reference_score
            material_change = {
                m: values['registered_complex_nrmse'] / reference['refined']['by_material'][m]['registered_complex_nrmse'] - 1.
                for m, values in metrics['refined']['by_material'].items()}
            accepted = gain >= .02 and max(material_change.values()) <= .02
            save_results(root, f'step_{steps:04d}', metrics, records)
            history.append({'step': steps, 'train_samples_seen': seen, 'val': metrics,
                            'relative_gain': gain, 'material_relative_changes': material_change,
                            'accepted': accepted, 'elapsed_seconds': time.monotonic() - started})
            if accepted and current < best_score:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                best_score = current
                selected_checkpoint = str((ckpt_dir / 'checkpoint_best.pt').resolve())
                torch.save({'experiment_version': VERSION, 'spatial_dims': 2, 'stage': 'refiner',
                            'epoch': 0, 'optimizer_steps': steps, 'config': config,
                            'model_state_dict': model.state_dict(), 'best_val_metric': current,
                            'checkpoint_selection_metric': 'registered_complex_nrmse',
                            'registered_protocol': {'objective': OBJECTIVE_VERSION, 'search_radius': 4.},
                            'validation_metrics': metrics, 'reference_metrics': reference,
                            'data_manifests': {s: d.manifest for s, d in datasets.items()},
                            'lineage': lineage, 'v10_protocol': protocol}, selected_checkpoint)
                save_results(root, 'best', metrics, records)
            (root / 'report.json').write_text(json.dumps({
                'reference_val': reference_score, 'best_accepted_val': best_score,
                'selected_checkpoint': selected_checkpoint, 'history': history,
                'warning': 'Validation selection only. Test improvement is not established.'},
                indent=2, allow_nan=False), encoding='utf-8')
            LOG.info('Step=%d | val=%.6f | gain=%.2f%% | worst_material_change=%.2f%% | accepted=%s',
                     steps, current, 100 * gain, 100 * max(material_change.values()), accepted)
            return accepted

        for raw in train_loader:
            if micro_count == 0:
                group_target = min(effective_batch, len(datasets['train']) - seen)
            batch = prepare_batch(raw, config, device)
            out = model(batch['input'], batch['diffraction'])
            loss, _ = objective_and_metrics(out.refined_object, batch['realspace'], 4.)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite training loss.')
            n = len(out.refined_object)
            (loss * (n / group_target)).backward()
            group_samples += n
            micro_count += 1
            seen += n
            if micro_count == args.accumulation_steps or seen == len(datasets['train']):
                if group_samples != group_target:
                    raise RuntimeError('Gradient accumulation accounting differs.')
                if steps == 0:
                    for name, module in (('phase', model.phase_model), ('refiner', model.refiner)):
                        grads = [p.grad for p in module.parameters() if p.grad is not None]
                        norm = torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads])) if grads else torch.tensor(0.)
                        if not bool(torch.isfinite(norm) & (norm > 0)):
                            raise RuntimeError(f'Missing or invalid {name} gradients.')
                        LOG.info('Gradient verified | %s_norm=%.6g', name, float(norm))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_count = group_samples = 0
                steps += 1
                if steps % 20 == 0:
                    LOG.info('TRAIN | update=%d/%d | samples=%d | loss=%.6f | elapsed=%.1fs',
                             steps, args.max_steps, seen, float(loss), time.monotonic() - started)
                if steps % args.check_every == 0 or steps == args.max_steps:
                    accepted = validate()
                    if not accepted:
                        stop_reason = 'validation_gate_not_met'
                        break
                if steps >= args.max_steps:
                    break
        if not history or history[-1]['step'] != steps:
            validate()
        LOG.info('DONE | reference=%.6f | best_accepted=%.6f | stop=%s', reference_score, best_score, stop_reason)
        LOG.info('Selected checkpoint=%s', selected_checkpoint)
        LOG.info('Source checkpoint preserved. No test evaluation performed.')
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == '__main__':
    main()
