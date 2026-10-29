"""Evaluate the frozen 100 samples with the original cascade: PSNR and MAE only.

Uses exactly the full-field COMPLEX metric function used during selection.
No training, sample selection, HIO, reconstruction timing, or image export.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = PROJECT_DIR / 'artifacts/sample_selection/original_train100_complex_v2_retry_b1/selected100'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-dir', type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument('--checkpoint', default='', help='Defaults to the frozen original checkpoint in the sample manifest.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive.')

    import h5py
    import numpy as np
    import torch
    from .data import prepare_batch
    from .train import load_checkpoint, make_model
    from .select_original_train100_complex import VERSION, complex_image_metrics, digest

    samples_dir = args.samples_dir.expanduser().resolve()
    # Verify transfer integrity before reading data or running inference.
    expected_names = {'selected_samples.h5', 'selected_samples.csv', 'selection_manifest.json'}
    checksums = {}
    for line in (samples_dir / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        checksum, filename = line.split(None, 1)
        filename = filename.strip()
        if filename not in expected_names or filename in checksums:
            raise ValueError('Unexpected or duplicate checksum entry.')
        checksums[filename] = checksum
    if set(checksums) != expected_names:
        raise ValueError('Incomplete sample bundle checksums.')
    for filename, checksum in checksums.items():
        if digest(samples_dir / filename) != checksum:
            raise ValueError(f'Checksum mismatch: {filename}')
    manifest = json.loads((samples_dir / 'selection_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != VERSION or manifest.get('num_samples') != 100:
        raise ValueError('Expected the frozen complex-metric 100-sample bundle.')
    thresholds = manifest.get('confirmed_precision_thresholds', {})
    if thresholds != {'complex_psnr_db_strict_gt': 20., 'complex_mae_strict_lt': .1}:
        raise ValueError('Unexpected metric protocol in the sample manifest.')
    checkpoint_path = Path(args.checkpoint or manifest['checkpoint']).expanduser().resolve()
    if digest(checkpoint_path) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the original model used to select these samples.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get('stage') != 'refiner':
        raise ValueError('A complete original phase + refiner checkpoint is required.')
    config = copy.deepcopy(checkpoint['config'])
    if config.get('shape') != [512, 512] or config.get('diffraction_kind') != 'intensity':
        raise ValueError('Expected the original 512x512 intensity-input model.')
    model = make_model('refiner', config, checkpoint=checkpoint).to(device).eval()

    root = args.output_dir or PROJECT_DIR / 'artifacts/evaluations' / f'original_selected100_{datetime.now():%Y%m%d_%H%M%S}'
    if root.exists():
        raise FileExistsError(f'Choose a fresh output directory: {root}')
    records = []
    with h5py.File(samples_dir / 'selected_samples.h5', 'r') as data:
        for name in ('observed_counts', 'exit_wave_real', 'exit_wave_imag'):
            if data[name].shape != (100, 512, 512) or data[name].dtype != np.dtype('float32'):
                raise ValueError(f'Invalid array shape or dtype: {name}')
        ids = data['display_id'].asstr()[:].tolist()
        source_ids = data['source_index'][:].tolist()
        if ids != [f'{i:04d}' for i in range(1, 101)] or len(set(source_ids)) != 100:
            raise ValueError('Expected 100 ordered display IDs and unique original sample indices.')
        expected = manifest['samples']
        if ids != [r['display_id'] for r in expected] or source_ids != [r['source_index'] for r in expected]:
            raise ValueError('HDF5 sample identities differ from the frozen manifest.')
        if not data['valid'][:].all():
            raise ValueError('The bundle contains invalid samples.')
        root.mkdir(parents=True)
        partial = root / 'per_sample.partial.csv'
        with partial.open('w', newline='', encoding='utf-8-sig') as handle, torch.no_grad():
            writer = csv.DictWriter(handle, fieldnames=['sample_id', 'PSNR_dB', 'MAE'])
            writer.writeheader()
            for start in range(0, 100, args.batch_size):
                stop = min(start + args.batch_size, 100)
                counts = data['observed_counts'][start:stop]
                real = data['exit_wave_real'][start:stop]
                imag = data['exit_wave_imag'][start:stop]
                raw = {'diffraction': torch.from_numpy(counts)[:, None],
                       'realspace': torch.complex(torch.from_numpy(real), torch.from_numpy(imag))[:, None]}
                batch = prepare_batch(raw, config, device)
                # Truth is used only for scoring, never as a model input.
                output = model(batch['input'], batch['diffraction'])
                values = complex_image_metrics(output.refined_object, batch['realspace'])
                psnr = values['complex_psnr_db'].cpu().tolist()
                mae = values['complex_mae'].cpu().tolist()
                for i in range(stop - start):
                    if not (math.isfinite(psnr[i]) and math.isfinite(mae[i])):
                        raise ValueError(f'Nonfinite score for sample {ids[start + i]}')
                    row = {'sample_id': ids[start + i], 'PSNR_dB': psnr[i], 'MAE': mae[i]}
                    records.append(row)
                    writer.writerow(row)
                    print(f"{row['sample_id']} | PSNR={row['PSNR_dB']:.4f} dB | MAE={row['MAE']:.6f}", flush=True)
                handle.flush()
        if len(records) != 100:
            raise RuntimeError('The run did not complete all 100 frozen samples.')
        partial.rename(root / 'per_sample.csv')
    summary = {'mean_PSNR_dB': float(np.mean([r['PSNR_dB'] for r in records])),
               'mean_MAE': float(np.mean([r['MAE'] for r in records]))}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(f"MEAN | PSNR={summary['mean_PSNR_dB']:.4f} dB | MAE={summary['mean_MAE']:.6f}", flush=True)
    print(f'Results: {root.resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
