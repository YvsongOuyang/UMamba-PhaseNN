"""Score the original cascade on TRAIN and freeze a performance-selected demo set.

This is explicitly a training-set demonstration selection, NOT an independent
test or a formal acceptance conclusion. No training, HIO, or timing assessment.
Selection uses full-field COMPLEX PSNR and MAE, including phase errors.
Confirmed strict thresholds: PSNR > 20 dB and MAE < 0.1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np

VERSION = 'original_training_complex_demo100_v2'
ORIGINAL = '/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/ecdi_2d512_20261014_224534_refiner/checkpoint_best.pt'
QUOTAS = {
    'graphite_anode': (12, 4, 1, 0),
    'licoo2_cathode': (12, 3, 1, 1),
    'lifepo4_cathode': (12, 3, 2, 0),
    'limn2o4_cathode': (12, 3, 1, 1),
    'lithium_metal_anode': (11, 3, 2, 0),
    'silicon_reference': (11, 4, 1, 0),
}
BANDS = [('preferred', 0., .20), ('good', .20, .40),
         ('middle', .40, .60), ('challenging', .70, .85)]
LOG = logging.getLogger(__name__)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def thresholds(row):
    psnr, mae = float(row['complex_psnr_db']), float(row['complex_mae'])
    return {'pass_complex_psnr20_mae010': int(psnr > 20 and mae < .1)}


def complex_image_metrics(prediction, target):
    """Full-image complex errors, not amplitude errors or a two-channel average.

    Reuse ORIGINAL COM/direct-twin/global-phase alignment and whole-image RMS
    normalization. Then divide BOTH aligned fields by the SAME truth amplitude
    maximum. The normalized truth has max modulus 1 (PSNR peak=1). Prediction
    is NOT clipped or independently peak-normalized. All 512x512 pixels count.
    Global phase/twin/translation and absolute energy scale are treated as
    ambiguities, as in the original evaluation; spatial phase errors remain.
    This explicit project convention is not claimed to be the only possible
    definition of complex PSNR/MAE.
    """
    import torch
    from .evaluate import aligned_objects_2d

    with torch.no_grad():
        pred, truth = aligned_objects_2d(prediction, target, threshold=.3)
        scale = truth.abs().amax(dim=(-2, -1), keepdim=True)
        if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError('Invalid truth normalization scale.')
        error = ((pred - truth) / scale).abs()
        # Double-precision accumulation does not change input fields.
        mae = error.to(torch.float64).flatten(1).mean(1)
        mse = error.to(torch.float64).square().flatten(1).mean(1)
        # Clamp for a finite CSV value only at numerical zero; never clip fields.
        psnr = -10 * mse.clamp_min(torch.finfo(mse.dtype).tiny).log10()
        return {'complex_psnr_db': psnr, 'complex_mae': mae,
                'complex_mse': mse, 'complex_rmse': mse.sqrt()}


def scan(args, root):
    import torch
    from .data import build_dataset, load_config, prepare_batch
    from .evaluate import metrics_2d
    from .train import build_loader, load_checkpoint, make_model, verify_data_identity

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get('stage') != 'refiner':
        raise ValueError('The original COMPLETE phase + refiner checkpoint is required.')
    config = load_config(args.data_config)
    if config.get('storage') != 'ecdi_hdf5':
        raise ValueError('This selector expects the original eCDI HDF5 dataset.')
    for key in ('input_log_data', 'center_targets', 'diffraction_kind'):
        if config[key] != checkpoint['config'][key]:
            raise ValueError(f'Checkpoint/config disagreement: {key}')
    config['root'] = str(Path(args.data_dir).resolve())
    config['training'] = checkpoint['config']['training']
    dataset = build_dataset(config, 'train', 0, include_ideal=True)
    try:
        verify_data_identity(checkpoint, dataset.manifest)
        if len(dataset) != dataset.manifest['available_samples']:
            raise ValueError('The entire immutable TRAIN split must be scored.')
        if len(dataset) < 100:
            raise ValueError('Insufficient training samples.')
        loader = build_loader(dataset, args.batch_size, args.num_workers, False, device)
        model = make_model('refiner', config, checkpoint=checkpoint).to(device).eval()
        checkpoint_hash = digest(args.checkpoint)
        LOG.info('Original complete cascade | checkpoint=%s | epoch=%s | sha256=%s',
                 args.checkpoint, checkpoint['epoch'], checkpoint_hash)
        LOG.info('Scoring all TRAIN samples=%d; no training and no test-set access.', len(dataset))
        partial = root / 'all_train_scores.partial.csv'
        seen = set()
        with partial.open('w', newline='', encoding='utf-8') as handle, torch.no_grad():
            writer = None
            for number, raw in enumerate(loader, 1):
                batch = prepare_batch(raw, config, device)
                out = model(batch['input'], batch['diffraction'])
                values = metrics_2d(out.refined_object, batch['realspace'], batch['diffraction'],
                                    batch['input'], .3, batch.get('ideal_intensity'))
                values.update(complex_image_metrics(out.refined_object, batch['realspace']))
                arrays = {k: v.detach().cpu().tolist() for k, v in values.items()}
                for i, sample in enumerate(raw['index']):
                    sid = int(sample)
                    if sid in seen:
                        raise ValueError(f'Duplicate training sample: {sid}')
                    seen.add(sid)
                    row = {'sample': sid, 'stage': 'refined', 'material': raw['material'][i],
                           **{k: v[i] for k, v in arrays.items()}}
                    if not all(math.isfinite(float(row[k])) for k in values):
                        raise ValueError(f'Nonfinite metrics for sample {sid}')
                    row.update(thresholds(row))
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                handle.flush()
                if number % 20 == 0 or number == len(loader):
                    LOG.info('SCORE | batch=%d/%d | samples=%d/%d', number, len(loader), len(seen), len(dataset))
        if seen != set(dataset.indices):
            raise ValueError('Scored sample IDs differ from the complete train manifest.')
        # Only a completed scan is published under the final CSV name.
        partial.rename(root / 'all_train_scores.csv')
        import inspect
        sources = [Path(__file__), Path(inspect.getsourcefile(metrics_2d)), Path(inspect.getsourcefile(make_model))]
        info = {'version': VERSION, 'split': 'train', 'num_scored': len(dataset),
                'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_epoch': checkpoint['epoch'],
                'checkpoint_sha256': checkpoint_hash, 'source_data': dataset.manifest,
                'data_config': config, 'seed': args.seed,
                'source_code_sha256': {str(p.resolve()): digest(p) for p in sources},
                'python': sys.version, 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'metric_scope': 'Final refined COMPLEX field, all 512x512 pixels, no cropping or masking in PSNR/MAE; original amplitude/phase metrics are diagnostics only.',
                'metric_protocol': 'Original evaluate.aligned_objects_2d: amplitude COM centering, MAE-selected direct/twin, support-fitted global phase, per-object full-field RMS normalization; then BOTH complex fields divided by the SAME aligned-truth amplitude maximum. Complex MSE=mean(abs(pred-truth)**2), complex MAE=mean(abs(pred-truth)), PSNR=-10*log10(MSE), peak modulus=1. No real/imaginary-channel averaging, no clipping, no predicted-peak normalization; all 512x512 pixels are scored.',
                'score_csv_sha256': digest(root / 'all_train_scores.csv')}
        write_json(root / 'scan_info.json', info)
    finally:
        dataset.close()


def ranked_material(rows):
    """Equal-weight within-material PSNR and MAE ranks; lower score is better."""
    n = len(rows)
    psnr_order = sorted(rows, key=lambda r: (-float(r['complex_psnr_db']), int(r['sample'])))
    mae_order = sorted(rows, key=lambda r: (float(r['complex_mae']), int(r['sample'])))
    psnr_rank = {int(r['sample']): i for i, r in enumerate(psnr_order)}
    mae_rank = {int(r['sample']): i for i, r in enumerate(mae_order)}
    result = []
    for original in rows:
        row = dict(original)
        sid = int(row['sample'])
        row['quality_rank_score'] = (psnr_rank[sid] + mae_rank[sid]) / (2 * max(n - 1, 1))
        result.append(row)
    result.sort(key=lambda r: (r['quality_rank_score'], -float(r['complex_psnr_db']), int(r['sample'])))
    for i, row in enumerate(result):
        row['quality_percentile_within_material'] = (i + .5) / n
    return result


def spread_pick(pool, count, selected_so_far, edges, rng):
    """Prefer underrepresented dose/thickness cells within each quality band."""
    def cell(row):
        return (int(np.searchsorted(edges[0], row['dose_e_per_a2'], side='right')),
                int(np.searchsorted(edges[1], row['thickness_a'], side='right')))
    occupancy = {}
    for row in selected_so_far:
        key = cell(row)
        occupancy[key] = occupancy.get(key, 0) + 1
    remaining, picked = list(pool), []
    if len(remaining) < count:
        raise ValueError('Quality band contains too few samples; no silent fallback is permitted.')
    for _ in range(count):
        available = {cell(row) for row in remaining}
        minimum = min(occupancy.get(key, 0) for key in available)
        eligible = [row for row in remaining if occupancy.get(cell(row), 0) == minimum]
        row = rng.choice(eligible)
        picked.append(row)
        remaining.remove(row)
        occupancy[cell(row)] = occupancy.get(cell(row), 0) + 1
    return picked


def select(args, root):
    import h5py
    with (root / 'scan_info.json').open(encoding='utf-8') as handle:
        info = json.load(handle)
    if info.get('version') != VERSION:
        raise ValueError('A new complete COMPLEX-metric scan is required; amplitude-only scores cannot be reused.')
    if info['split'] != 'train' or digest(root / 'all_train_scores.csv') != info['score_csv_sha256']:
        raise ValueError('Missing TRAIN provenance or scan CSV checksum mismatch.')
    with (root / 'all_train_scores.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != info['num_scored'] or len({int(r['sample']) for r in rows}) != len(rows):
        raise ValueError('Training scores are incomplete or duplicated.')
    if {r['material'] for r in rows} != set(QUOTAS):
        raise ValueError('Expected exactly the six original materials.')
    for row in rows:
        if row['stage'] != 'refined':
            raise ValueError('Selection must use final refined metrics.')
        if not all(math.isfinite(float(row[k])) for k in ('complex_psnr_db', 'complex_mae')):
            raise ValueError('Nonfinite selection metric.')
    bundle = root / 'selected100'
    if bundle.exists():
        raise FileExistsError('The selected set is already frozen; use a new output directory to create another version.')
    source_path = Path(info['source_data']['dataset_path'])
    rng = random.Random(info['seed'])
    selected = []
    with h5py.File(source_path, 'r') as source:
        if str(source.attrs.get('config_sha256', '')) != info['source_data']['config_sha256']:
            raise ValueError('Source dataset configuration changed since scanning.')
        # Check exact membership in the immutable TRAIN manifest before exporting.
        manifest = json.loads(Path(info['source_data']['manifest_path']).read_text(encoding='utf-8'))
        if digest(info['source_data']['manifest_path']) != info['source_data']['manifest_sha256']:
            raise ValueError('Source split manifest changed.')
        if set(manifest['splits']['train']) != {int(r['sample']) for r in rows}:
            raise ValueError('Score IDs are not exactly the immutable TRAIN split.')
        for row in rows:
            md = json.loads(source['metadata_json'].asstr()[int(row['sample'])])
            if md['structure']['source_name'] != row['material']:
                raise ValueError('Material metadata mismatch.')
            row['dose_e_per_a2'] = float(md['pulse']['dose_e_per_a2'])
            row['thickness_a'] = float(md['structure']['target_thickness_a'])
        for material, quota in QUOTAS.items():
            ranked = ranked_material([r for r in rows if r['material'] == material])
            edges = (np.quantile([r['dose_e_per_a2'] for r in ranked], [1 / 3, 2 / 3]),
                     np.quantile([r['thickness_a'] for r in ranked], [1 / 3, 2 / 3]))
            material_selected = []
            for (name, low, high), count in zip(BANDS, quota):
                if not count:
                    continue
                pool = [r for r in ranked if low <= r['quality_percentile_within_material'] < high]
                chosen = spread_pick(pool, count, material_selected, edges, rng)
                for row in chosen:
                    row['selection_band'] = name
                material_selected.extend(chosen)
            selected.extend(material_selected)
        if len(selected) != 100 or len({int(r['sample']) for r in selected}) != 100:
            raise ValueError('Selection must contain exactly 100 unique samples.')
        rng.shuffle(selected)
        selected = [{'display_id': f'{i:04d}', 'source_split': 'train', **r}
                    for i, r in enumerate(selected, 1)]
        bundle.mkdir()
        # Export RAW input and RAW complex truth; scoring transforms are never baked into data.
        partial = bundle / 'selected_samples.partial.h5'
        with h5py.File(partial, 'w') as target:
            for name in ('observed_counts', 'exit_wave_real', 'exit_wave_imag', 'ideal_intensity'):
                target.create_dataset(name, shape=(100, 512, 512), dtype=source[name].dtype,
                                      chunks=(1, 512, 512), compression='gzip', compression_opts=4, shuffle=True)
            target.create_dataset('source_index', data=np.array([int(r['sample']) for r in selected], dtype=np.int64))
            text = h5py.string_dtype('utf-8')
            target.create_dataset('display_id', data=np.array([r['display_id'] for r in selected], dtype=object), dtype=text)
            target.create_dataset('metadata_json', shape=(100,), dtype=text)
            target.create_dataset('valid', data=np.ones(100, dtype=bool))
            for i, row in enumerate(selected):
                sid = int(row['sample'])
                if not bool(source['valid'][sid]):
                    raise ValueError(f'Source sample became invalid: {sid}')
                for name in ('observed_counts', 'exit_wave_real', 'exit_wave_imag', 'ideal_intensity'):
                    target[name][i] = source[name][sid]
                target['metadata_json'][i] = source['metadata_json'].asstr()[sid]
            target.attrs['selection_version'] = VERSION
            target.attrs['purpose'] = 'performance_selected_training_demonstration'
            target.attrs['source_split'] = 'train'
            target.attrs['source_dataset_config_sha256'] = info['source_data']['config_sha256']
            target.attrs['model_sha256'] = info['checkpoint_sha256']
        partial.rename(bundle / 'selected_samples.h5')
    with (bundle / 'selected_samples.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        'version': VERSION, 'num_samples': 100, 'source_split': 'train',
        'is_independent_test_set': False, 'formal_acceptance_conclusion': None,
        'purpose': 'Performance-selected training-set demonstration; not an unbiased generalization assessment.',
        'selection_seed': info['seed'], 'checkpoint': info['checkpoint'], 'checkpoint_sha256': info['checkpoint_sha256'],
        'scan_info_sha256': digest(root / 'scan_info.json'),
        'source_dataset_identity': info['source_data'],
        'metric_protocol': info['metric_protocol'], 'metric_scope': info['metric_scope'],
        'ranking': 'Within each material: equal-weight rank of descending COMPLEX PSNR and ascending COMPLEX MAE; then ascending combined rank.',
        'quality_bands': {name: {'percentile_interval': [low, high], 'selected': sum(r['selection_band'] == name for r in selected)}
                          for name, low, high in BANDS},
        'diversity': 'Prefer underrepresented dose/thickness tertile cells within each material; deterministic seeded draws.',
        'material_counts': {m: sum(r['material'] == m for r in selected) for m in QUOTAS},
        'mean_complex_psnr_db': float(np.mean([float(r['complex_psnr_db']) for r in selected])),
        'mean_complex_mae': float(np.mean([float(r['complex_mae']) for r in selected])),
        'confirmed_precision_thresholds': {'complex_psnr_db_strict_gt': 20., 'complex_mae_strict_lt': .1},
        'pass_count_complex_psnr20_mae010': sum(thresholds(r)['pass_complex_psnr20_mae010'] for r in selected),
        'timing_assessed': False, 'traditional_HIO_assessed': False,
        'confirmed_document_resolutions': ['User confirmed strict PSNR > 20dB AND MAE < 0.1.',
                                          'User confirmed 100 samples from TRAIN and the original model.',
                                          'User confirmed full-field complex evaluation, including amplitude and phase.',
                                          'This set is performance-selected; no independent-test claim.'],
        'samples': [{'display_id': r['display_id'], 'source_index': int(r['sample']),
                     'material': r['material'], 'selection_band': r['selection_band']} for r in selected],
    }
    summary['files_sha256'] = {name: digest(bundle / name) for name in ('selected_samples.h5', 'selected_samples.csv')}
    write_json(bundle / 'selection_manifest.json', summary)
    checksums = {**summary['files_sha256'], 'selection_manifest.json': digest(bundle / 'selection_manifest.json')}
    (bundle / 'SHA256SUMS.txt').write_text(''.join(f'{value}  {name}\n' for name, value in checksums.items()), encoding='utf-8')
    LOG.info('SELECTED | count=100 | bands=70 preferred + 20 good + 8 middle + 2 challenging')
    LOG.info('Screening results only | mean COMPLEX PSNR=%.4f | mean COMPLEX MAE=%.6f',
             summary['mean_complex_psnr_db'], summary['mean_complex_mae'])
    LOG.info('COMPLEX precision screening | PSNR > 20dB AND MAE < 0.1 | count=%d/100',
             summary['pass_count_complex_psnr20_mae010'])
    LOG.info('DONE | selected bundle=%s', bundle.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=ORIGINAL)
    parser.add_argument('--data-config', default=str(Path(__file__).parent / 'configs/ecdi_2d.json'))
    parser.add_argument('--data-dir', default='/data_ssd/oyys/eCDI')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/sample_selection/original_train100_complex_v2'))
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--stage', choices=['all', 'scan', 'select'], default='all')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Batch size must be positive; workers must be nonnegative.')
    root = args.output_dir
    if args.stage != 'select':
        if root.exists():
            raise FileExistsError('Choose a fresh output directory. A completed scan can be selected with --stage select.')
        root.mkdir(parents=True)
    elif not (root / 'scan_info.json').is_file():
        raise FileNotFoundError('Selection requires a complete scan_info.json and all_train_scores.csv.')
    logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(root / f'{args.stage}.log', mode='x')])
    if args.stage in ('all', 'scan'):
        scan(args, root)
    if args.stage in ('all', 'select'):
        select(args, root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
