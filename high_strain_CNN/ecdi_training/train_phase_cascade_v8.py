"""Optimize phase through the fixed V7 refiner using final registered error.

Native checkpoints contain the COMPLETE cascade (stage='refiner'), although
only phase parameters are optimized. Refiner weights and buffers are frozen.
Phase BatchNorm running statistics are frozen for microbatch accumulation;
phase convolutions and BatchNorm affine parameters remain trainable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_v2 import build_dataset, prepare_batch
from .train import load_checkpoint, make_model, verify_data_identity
from .aligned_objective_v7 import OBJECTIVE_VERSION, objective_and_metrics
from .train_aligned_v7 import (
    evaluate, file_digest, read_thickness, restore_rng, rng_state,
    save_results, state_digest,
)

VERSION = 'phase_through_fixed_refiner_v8'
PROJECT_DIR = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def phase_bn_digest(model):
    digest = hashlib.sha256()
    for name, module in model.phase_model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for key, tensor in module.named_buffers(recurse=False):
                if tensor is not None:
                    value = tensor.detach().cpu().contiguous()
                    digest.update(f'{name}:{key}:{value.dtype}:{tuple(value.shape)}'.encode())
                    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def configure_trainable(model):
    # The existing cascade otherwise wraps phase forward in torch.no_grad().
    model.freeze_phase_model = False
    for parameter in model.phase_model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.refiner.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.refiner.eval()


def train_epoch(model, loader, config, device, optimizer, radius, accumulation):
    model.train()
    model.refiner.eval()
    for module in model.phase_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    if model.freeze_phase_model or any(p.requires_grad for p in model.refiner.parameters()):
        raise RuntimeError('V8 requires trainable phase and frozen refiner parameters.')
    if any(m.training for m in model.refiner.modules()):
        raise RuntimeError('All refiner modules must be in eval mode.')
    parameters = [p for p in model.phase_model.parameters() if p.requires_grad]
    total_samples = len(loader.dataset)
    effective_batch = loader.batch_size * accumulation
    seen = micro_count = group_samples = optimizer_steps = 0
    totals = {}
    started = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for index, raw in enumerate(loader, 1):
        if micro_count == 0:
            optimizer.zero_grad(set_to_none=True)
            group_target = min(effective_batch, total_samples - seen)
        batch = prepare_batch(raw, config, device)
        output = model(batch['input'], batch['diffraction'])
        if not output.initial_object.requires_grad or not output.refined_object.requires_grad:
            raise RuntimeError('Gradient path from final output back to phase is disabled.')
        loss, values = objective_and_metrics(
            output.refined_object, batch['realspace'], radius
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite final registered loss.')
        n = len(output.refined_object)
        # Correct weighting also for the last incomplete effective batch.
        (loss * (n / group_target)).backward()
        micro_count += 1
        group_samples += n
        seen += n
        for key, value in {'loss': loss.detach(), **{k: v.mean() for k, v in values.items()}}.items():
            totals[key] = totals.get(key, 0.) + float(value) * n
        if micro_count == accumulation or seen == total_samples:
            if group_samples != group_target:
                raise RuntimeError('Gradient accumulation sample accounting differs.')
            active = [p for p in parameters if p.grad is not None]
            if not active or any(not bool(torch.isfinite(p.grad).all()) for p in active):
                raise RuntimeError('Missing or nonfinite phase gradients.')
            if any(p.grad is not None for p in model.refiner.parameters()):
                raise RuntimeError('Frozen refiner unexpectedly has parameter gradients.')
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad.detach()) for p in active
            ]))
            if optimizer_steps == 0:
                if not bool(norm > 0):
                    raise RuntimeError('Phase gradient is zero on the first effective batch.')
                LOGGER.info('Gradient path verified | phase_grad_norm=%.6g | refiner_parameter_grads=none', float(norm))
            clip = config['training'].get('gradient_clip', 1.)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(active, clip, error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            micro_count = group_samples = 0
        if index % 150 == 0 or index == len(loader):
            elapsed = time.monotonic() - started
            LOGGER.info('TRAIN | microbatch=%d/%d | steps=%d | samples=%d/%d | final_loss=%.6f | elapsed=%.1fs | eta=%.1fs',
                        index, len(loader), optimizer_steps, seen, total_samples, float(loss), elapsed,
                        elapsed / index * (len(loader) - index))
    if seen != total_samples or optimizer_steps != math.ceil(total_samples / effective_batch):
        raise RuntimeError('Training split or optimizer step count is incomplete.')
    result = {key: value / seen for key, value in totals.items()}
    result.update(samples=seen, optimizer_steps=optimizer_steps,
                  elapsed_seconds=time.monotonic() - started)
    if device.type == 'cuda':
        result['peak_memory_gib'] = torch.cuda.max_memory_allocated(device) / 1024**3
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-cascade-checkpoint')
    source.add_argument('--resume')
    parser.add_argument('--data-dir', default='/data_ssd/oyys/eCDI')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--stop-after-epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--search-radius', type=float, default=4.)
    parser.add_argument('--batch-size', type=int, default=4, help='Training microbatch size.')
    parser.add_argument('--accumulation-steps', type=int, default=3)
    parser.add_argument('--eval-batch-size', type=int, default=12)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-name', default='')
    parser.add_argument('--checkpoint-root', type=Path, default=Path('/data_ssd/oyys/high_strain_cnn/pipeline_output_2d'))
    parser.add_argument('--runs-dir', type=Path, default=PROJECT_DIR / 'artifacts/training/ecdi_training')
    args = parser.parse_args()
    if not 1 <= args.stop_after_epochs <= args.epochs <= 3:
        parser.error('Require 1 <= stop-after-epochs <= epochs <= 3.')
    if min(args.batch_size, args.accumulation_steps, args.eval_batch_size) < 1 or args.num_workers < 0:
        parser.error('Batch sizes and accumulation must be positive; workers nonnegative.')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 1e-6:
        parser.error('Require finite learning-rate > 1e-6.')
    if not math.isfinite(args.search_radius) or not 0 < args.search_radius <= 8:
        parser.error('Require finite 0 < search-radius <= 8.')
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    path = args.resume or args.init_cascade_checkpoint
    source = load_checkpoint(path)
    if source.get('stage') != 'refiner':
        raise ValueError('Initialization must contain the COMPLETE native cascade.')
    if args.init_cascade_checkpoint and source.get('experiment_version') != 'registered_refiner_v7':
        raise ValueError('Initialize from the selected V7 cascade, not a standalone phase or older refiner.')
    config = copy.deepcopy(source['config'])
    if (config.get('storage') != 'ecdi_hdf5' or list(config.get('shape', [])) != [512, 512]
            or config['training'].get('phase_architecture', 'baseline') != 'baseline'):
        raise ValueError('V8 expects the native 512x512 eCDI baseline cascade.')
    config['root'] = str(Path(args.data_dir).expanduser().resolve())
    protocol = {'epochs': args.epochs, 'learning_rate': args.learning_rate,
                'search_radius': args.search_radius, 'batch_size': args.batch_size,
                'accumulation_steps': args.accumulation_steps, 'eval_batch_size': args.eval_batch_size,
                'num_workers': args.num_workers, 'seed': args.seed, 'device': args.device,
                'objective': OBJECTIVE_VERSION, 'optimized_parameters': 'phase_only',
                'refiner_mode': 'frozen_eval_with_input_gradients', 'phase_bn_running_stats': 'frozen'}
    if args.resume and (source.get('experiment_version') != VERSION or source.get('v8_protocol') != protocol):
        raise ValueError('Resume requires the same V8 training protocol.')
    if source.get('registered_protocol', {}).get('search_radius') != args.search_radius:
        raise ValueError('Use the selected source checkpoint registration radius.')
    name = args.run_name or f'phase_cascade_v8_{datetime.now():%Y%m%d_%H%M%S}'
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Run name must be one directory name.')
    root, checkpoint_dir = args.runs_dir / name, args.checkpoint_root / name
    if args.resume:
        if Path(path).resolve().parent != checkpoint_dir.resolve() or not root.is_dir():
            raise ValueError('Resume requires the original run name and paths.')
        if source['epoch'] >= args.stop_after_epochs:
            raise ValueError('Stop epoch must exceed the saved epoch.')
    elif root.exists() or checkpoint_dir.exists():
        raise FileExistsError('Choose a fresh run name to preserve previous results.')
    datasets, loaders, generators = {}, {}, {}
    try:
        for split in ('train', 'val'):
            datasets[split] = build_dataset(config, split, 0, include_ideal=split == 'val')
            verify_data_identity(source, datasets[split].manifest)
            for key in ('config_sha256', 'manifest_sha256'):
                if not source.get('data_manifests', {}).get('train', {}).get(key):
                    raise ValueError('Source checkpoint lacks dataset identity hashes.')
            actual = datasets[split].manifest
            previous = source.get('data_manifests', {}).get(split, {})
            if actual['selected_samples'] != actual['available_samples']:
                raise ValueError('Use the complete immutable train and validation splits.')
            if previous.get('selected_indices_sha256') != actual['selected_indices_sha256']:
                raise ValueError('Source/current selected sample identities differ.')
            generators[split] = torch.Generator().manual_seed(args.seed + (split != 'train'))
            loaders[split] = DataLoader(datasets[split], batch_size=args.batch_size if split == 'train' else args.eval_batch_size,
                shuffle=split == 'train', num_workers=args.num_workers, generator=generators[split],
                persistent_workers=False, pin_memory=device.type == 'cuda', drop_last=False)
        thickness = read_thickness(datasets['val'])
        model = make_model('refiner', config, checkpoint=source).to(device)
        configure_trainable(model)
        refiner_hash, bn_hash = state_digest(model.refiner), phase_bn_digest(model)
        if args.resume and (source['frozen_refiner_sha256'] != refiner_hash or source['phase_bn_sha256'] != bn_hash):
            raise ValueError('Frozen model state differs from saved V8 state.')
        root.mkdir(parents=True, exist_ok=bool(args.resume))
        checkpoint_dir.mkdir(parents=True, exist_ok=bool(args.resume))
        logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler(root / 'console.log', encoding='utf-8')])
        optimizer = torch.optim.Adam(model.phase_model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        if args.resume:
            optimizer.load_state_dict(source['optimizer_state_dict'])
            scheduler.load_state_dict(source['scheduler_state_dict'])
            history, best = copy.deepcopy(source['history']), source['best_val_metric']
            reference, lineage = source['reference_metrics'], source['lineage']
            restore_rng(source['random_states'], generators)
            start = source['epoch'] + 1
        else:
            history, best, reference, start = [], float('inf'), None, 1
            lineage = {'source_checkpoint': str(Path(path).resolve()), 'source_sha256': file_digest(path),
                       'source_epoch': source['epoch'], 'source_best_val_metric': source.get('best_val_metric')}
        config['experiment_version'] = VERSION
        config['training']['phase'] = {'objective': OBJECTIVE_VERSION, 'optimization_target': 'final_refined_object',
            'epochs': args.epochs, 'learning_rate': args.learning_rate, 'scheduler': 'cosine', 'batchnorm_running_stats': 'frozen'}
        manifests = {split: dataset.manifest for split, dataset in datasets.items()}
        (root / 'config.json').write_text(json.dumps({'config': config, 'protocol': protocol, 'lineage': lineage,
            'data_manifests': manifests}, indent=2), encoding='utf-8')
        LOGGER.info('Experiment=%s | trainable=phase only | fixed=refiner and phase BN statistics', VERSION)
        LOGGER.info('Objective=%s on FINAL output | microbatch=%d | accumulation=%d | effective_batch=%d | val_batch=%d',
                    OBJECTIVE_VERSION, args.batch_size, args.accumulation_steps,
                    args.batch_size * args.accumulation_steps, args.eval_batch_size)
        LOGGER.info('Source epoch=%s | run=%s | checkpoints=%s', lineage['source_epoch'], root, checkpoint_dir)

        def check_frozen():
            if state_digest(model.refiner) != refiner_hash:
                raise RuntimeError('Frozen refiner weights or buffers changed.')
            if phase_bn_digest(model) != bn_hash:
                raise RuntimeError('Frozen phase BatchNorm statistics changed.')

        def validate_save(epoch, train=None):
            nonlocal best, reference
            check_frozen()
            summary, records = evaluate(model, loaders['val'], config, device, thickness, args.search_radius)
            check_frozen()
            values = summary['refined']['all']
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError('Nonfinite validation metric.')
            metric = values['registered_complex_nrmse']
            if reference is None:
                reference = copy.deepcopy(summary)
                save_results(root, 'reference', summary, records)
            improved = metric < best
            best = min(best, metric)
            history.append({'epoch': epoch, 'train': train, 'val': summary, 'lr': optimizer.param_groups[0]['lr']})
            saved = {'experiment_version': VERSION, 'spatial_dims': 2, 'stage': 'refiner', 'epoch': epoch,
                'config': config, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'best_val_metric': best,
                'checkpoint_selection_metric': 'registered_complex_nrmse', 'v8_protocol': protocol,
                'registered_protocol': {'objective': OBJECTIVE_VERSION, 'search_radius': args.search_radius},
                'reference_metrics': reference, 'validation_metrics': summary, 'lineage': lineage, 'history': history,
                'data_manifests': manifests, 'frozen_refiner_sha256': refiner_hash, 'phase_bn_sha256': bn_hash,
                'random_states': rng_state(generators)}
            torch.save(saved, checkpoint_dir / 'checkpoint_last.pt')
            if improved:
                torch.save(saved, checkpoint_dir / 'checkpoint_best.pt')
                save_results(root, 'best', summary, records)
            save_results(root, 'last', summary, records)
            (root / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False), encoding='utf-8')
            LOGGER.info('Epoch=%d/%d | final_registered_nrmse=%.6f | best=%.6f | reference=%.6f | initial_nrmse=%.6f | amplitude_rms_error=%.6f | thick_nrmse=%.6f | offaxis_nrmse=%.6f | boundary_fraction=%.3f | lr=%.3g',
                epoch, args.epochs, metric, best, reference['refined']['all']['registered_complex_nrmse'],
                summary['initial']['all']['registered_complex_nrmse'], values['amplitude_relative_l2_rms'],
                summary['refined']['thick_ge34'].get('registered_complex_nrmse', float('nan')),
                values['offaxis_nrmse_9x9'], values['search_boundary_hit'], optimizer.param_groups[0]['lr'])
        if not args.resume:
            validate_save(0)
        for epoch in range(start, args.stop_after_epochs + 1):
            train = train_epoch(model, loaders['train'], config, device, optimizer, args.search_radius, args.accumulation_steps)
            check_frozen()
            scheduler.step()
            validate_save(epoch, train)
        LOGGER.info('Pilot stopped | best=%.6f | reference=%.6f | source checkpoint preserved=%s',
                    best, reference['refined']['all']['registered_complex_nrmse'], lineage['source_checkpoint'])
    finally:
        for dataset in datasets.values():
            dataset.close()
    return 0


if __name__ == '__main__':
    main()
