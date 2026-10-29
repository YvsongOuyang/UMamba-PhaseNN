"""Bounded fitting diagnostic on TRAIN samples; never selects a production model.

Both cascade stages can learn, with dropout disabled and BN statistics fixed.
Validation is read only. No checkpoint is saved or overwritten.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .aligned_objective_v7 import objective_and_metrics
from .data_v2 import build_dataset, prepare_batch
from .train import load_checkpoint, make_model, verify_data_identity

LOG = logging.getLogger(__name__)


def metadata(dataset):
    import h5py
    with h5py.File(dataset.path, 'r') as source:
        values = source['metadata_json'].asstr()
        return {int(i): json.loads(values[i])['structure'] for i in dataset.indices}


@torch.no_grad()
def score(model, batches, device, config):
    model.eval()
    rows = []
    for raw in batches:
        batch = prepare_batch(raw, config, device)
        out = model(batch['input'], batch['diffraction'])
        _, metrics = objective_and_metrics(out.refined_object, batch['realspace'], 4.)
        arrays = {k: v.cpu().tolist() for k, v in metrics.items()}
        for i, sid in enumerate(raw['index']):
            rows.append({'sample': int(sid), 'material': raw['material'][i],
                         **{k: v[i] for k, v in arrays.items()}})
    return rows


def summary(rows):
    key = 'registered_complex_nrmse'
    return {'n': len(rows), 'nrmse': float(np.mean([r[key] for r in rows])),
            'by_material': {m: float(np.mean([r[key] for r in rows if r['material'] == m]))
                            for m in sorted({r['material'] for r in rows})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='/data_ssd/oyys/eCDI')
    parser.add_argument('--output-dir', default='artifacts/evaluations/fit_diagnosis_v9')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--max-seconds', type=float, default=180.)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if (args.steps < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0
            or args.batch_size < 1 or args.num_workers < 0):
        parser.error('Steps, time and batch size must be positive; workers nonnegative.')
    root = Path(args.output_dir)
    if root.exists():
        raise FileExistsError(f'Choose a new output directory: {root}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    source = load_checkpoint(args.checkpoint)
    if source.get('stage') != 'refiner':
        raise ValueError('A complete cascade checkpoint is required.')
    if source.get('registered_protocol', {}).get('search_radius') != 4.:
        raise ValueError('Expected registration radius=4, as in V7/V8.')
    config = copy.deepcopy(source['config'])
    config['root'] = str(Path(args.data_dir).resolve())
    datasets = {}
    root.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, force=True,
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(root / 'console.log')])
    try:
        for split in ('train', 'val'):
            datasets[split] = build_dataset(config, split, include_ideal=True)
            verify_data_identity(source, datasets[split].manifest)
        train = datasets['train']
        md = metadata(train)
        materials = sorted({m['source_name'] for m in md.values()})
        if len(materials) != 6:
            raise ValueError('Expected six materials.')
        # Candidates are chosen ONLY from the immutable training split.
        candidates = []
        rng = random.Random(42)
        for material in materials:
            ids = [i for i in train.indices if md[i]['source_name'] == material]
            for low, high in ((0., 20.), (34., float('inf'))):
                group = [i for i in ids if low <= md[i]['target_thickness_a'] <= high]
                if len(group) < 2:
                    raise ValueError('Insufficient training candidates in a thickness group.')
                candidates.extend(rng.sample(group, 2))
        positions = {sid: i for i, sid in enumerate(train.indices)}

        def loader(dataset, indices):
            return DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, drop_last=False)

        model = make_model('refiner', config, checkpoint=source).to(device).eval()
        candidate_rows = score(model, loader(train, [positions[i] for i in candidates]), device, config)
        selected = []
        for material in materials:
            rows = sorted((r for r in candidate_rows if r['material'] == material),
                          key=lambda r: r['registered_complex_nrmse'], reverse=True)
            selected.extend(r['sample'] for r in rows[:2])
        selected_loader = loader(train, [positions[i] for i in selected])
        # Tiny selected training set remains cached; no repeated HDF5 reads in fitting.
        raw_batches = list(selected_loader)
        cached = [prepare_batch(raw, config, device) for raw in raw_batches]
        val_loader = loader(datasets['val'], list(range(min(48, len(datasets['val'])))))
        train_before = summary([r for r in candidate_rows if r['sample'] in selected])
        val_before_rows = score(model, val_loader, device, config)
        if len({r['material'] for r in val_before_rows}) != 6:
            raise ValueError('The diagnostic validation subset must cover all six materials.')
        val_before = summary(val_before_rows)
        model.freeze_phase_model = False
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        # eval() disables stochastic dropout / BN updates, but DOES NOT disable autograd.
        model.eval()
        optimizer = torch.optim.Adam([
            {'params': model.phase_model.parameters(), 'lr': 1e-5},
            {'params': model.refiner.parameters(), 'lr': 1e-4}])
        LOG.info('TRAIN samples=12, two per material | joint fitting | BN fixed | dropout off')
        LOG.info('Before | train_nrmse=%.6f | val48_nrmse=%.6f', train_before['nrmse'], val_before['nrmse'])
        history = []
        started = time.monotonic()
        stop_reason = 'step_limit'
        for step in range(1, args.steps + 1):
            batch = cached[(step - 1) % len(cached)]
            optimizer.zero_grad(set_to_none=True)
            out = model(batch['input'], batch['diffraction'])
            loss, _ = objective_and_metrics(out.refined_object, batch['realspace'], 4.)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite loss.')
            loss.backward()
            if step == 1:
                for name, module in (('phase', model.phase_model), ('refiner', model.refiner)):
                    active = [p.grad for p in module.parameters() if p.grad is not None]
                    norm = torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in active])) if active else torch.tensor(0.)
                    if not bool(torch.isfinite(norm) & (norm > 0)):
                        raise RuntimeError(f'Missing or invalid {name} gradients.')
                    LOG.info('Gradient verified | %s_norm=%.6g', name, float(norm))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            timeout = time.monotonic() - started >= args.max_seconds
            if step % 20 == 0 or step == args.steps or timeout:
                current = summary(score(model, raw_batches, device, config))
                reduction = 1. - current['nrmse'] / train_before['nrmse']
                history.append({'step': step, 'train': current, 'relative_reduction': reduction})
                LOG.info('Step=%d/%d | train_nrmse=%.6f | reduction=%.1f%% | elapsed=%.1fs',
                         step, args.steps, current['nrmse'], 100 * reduction, time.monotonic() - started)
                if reduction >= .5:
                    stop_reason = 'train_error_halved'
                    break
                if timeout:
                    stop_reason = 'time_limit'
                    break
        train_after_rows = score(model, raw_batches, device, config)
        val_after_rows = score(model, val_loader, device, config)
        train_after, val_after = summary(train_after_rows), summary(val_after_rows)
        report = {'diagnostic_only': True, 'production_checkpoint_saved': False,
                  'checkpoint': str(Path(args.checkpoint).resolve()), 'source_epoch': source['epoch'],
                  'arguments': vars(args), 'selected_train_ids': selected,
                  'candidate_train_ids': candidates, 'stop_reason': stop_reason,
                  'train_before': train_before, 'train_after': train_after,
                  'val48_before': val_before, 'val48_after': val_after, 'history': history,
                  'warning': 'Finite-budget subset fitting diagnoses trainability, not test improvement or an architecture limit.'}
        (root / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        all_rows = []
        for tag, rows in [('train_before', [r for r in candidate_rows if r['sample'] in selected]),
                          ('train_after', train_after_rows), ('val_before', val_before_rows), ('val_after', val_after_rows)]:
            all_rows.extend({'evaluation': tag, **r} for r in rows)
        with (root / 'per_sample.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        LOG.info('DONE | train=%.6f -> %.6f | val48=%.6f -> %.6f | reason=%s',
                 train_before['nrmse'], train_after['nrmse'], val_before['nrmse'], val_after['nrmse'], stop_reason)
        LOG.info('Diagnostic only; source model unchanged. Results=%s', root.resolve())
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == '__main__':
    main()
