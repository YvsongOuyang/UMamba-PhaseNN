"""Compare original and Au5nm-finetuned models on the same frozen 100 examples."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from .benchmark_selected100 import preflight
from .data import prepare_batch
from .select_original_train100_complex import digest
from .periodic_metrics import METRIC_PROTOCOL, complex_image_metrics
from .hio_fixed import make_model_hio_starts
from .train import PROJECT_DIR, load_checkpoint, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-dir', type=Path,
                        default=Path('/data_ssd/oyys/eCDI/test_data/ecdi_100'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        '/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/'
        'au5nm_from_original_1029/checkpoint_best.pt'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--hio-method', choices=('fixed', 'shrinkwrap'), default='fixed',
                        help='Use the same fixed HIO/ER comparison as test_au5nm by default.')
    args = parser.parse_args()
    if args.hio_method == 'fixed':
        from .hio_fixed import HIO_DEFAULTS, reconstruct_hio
    else:
        from .benchmark_selected100 import HIO_DEFAULTS, reconstruct_hio
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu if needed.')
    folder = args.samples_dir.expanduser().resolve()
    manifest, _ = preflight(folder)
    original_path = Path(manifest['checkpoint']).expanduser().resolve()
    if digest(original_path) != manifest['checkpoint_sha256']:
        raise ValueError('Original checkpoint does not match the frozen sample manifest.')
    root = args.output_dir or PROJECT_DIR / 'artifacts/evaluations' / (
        f'au5nm_on_original100_periodic_{args.hio_method}_{datetime.now():%Y%m%d_%H%M%S}')
    root = root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f'Use a new output directory: {root}')
    paths = {'original': original_path, 'finetuned': args.checkpoint.expanduser().resolve()}
    models = {}
    baseline_config = None
    print(f'HIO method={args.hio_method} | parameters={HIO_DEFAULTS}', flush=True)
    with h5py.File(folder / 'ecdi_100_samples.h5', 'r') as data:
        ids = data['display_id'].asstr()[:].tolist()
        for label, path in paths.items():
            torch.manual_seed(42)
            checkpoint = load_checkpoint(path)
            if checkpoint.get('stage') != 'refiner':
                raise ValueError(f'{label}: a complete phase + refiner checkpoint is required.')
            config = copy.deepcopy(checkpoint['config'])
            if config.get('shape') != [512, 512] or config.get('diffraction_kind') != 'intensity':
                raise ValueError(f'{label}: expected a 512x512 intensity-input model.')
            if baseline_config is None:
                baseline_config = config
            for key in ('input_log_data', 'center_targets', 'diffraction_kind'):
                if config[key] != baseline_config[key]:
                    raise ValueError(f'Model preprocessing differs on {key}.')
            models[label] = make_model('refiner', config, checkpoint=checkpoint).to(device).eval()
            del checkpoint
        all_labels = ('original', 'finetuned', 'hio', 'model_hio')
        results = {label: [] for label in all_labels}
        recon_dir = root / 'reconstructions'
        root.mkdir(parents=True)
        recon_dir.mkdir()
        # Every method sees the same measurement and fixed HIO support. The
        # model-initialized trajectory starts from the trained fine-tuned model.
        with torch.no_grad():
            for i in range(100):
                counts = torch.from_numpy(data['observed_counts'][i:i+1])[:, None]
                real = torch.from_numpy(data['exit_wave_real'][i:i+1])[:, None]
                imag = torch.from_numpy(data['exit_wave_imag'][i:i+1])[:, None]
                batch = prepare_batch({'diffraction': counts,
                                       'realspace': torch.complex(real, imag)}, baseline_config, device)
                predictions = {
                    label: models[label](batch['input'], batch['diffraction']).refined_object
                    for label in ('original', 'finetuned')
                }
                predictions['hio'] = reconstruct_hio(
                    batch['diffraction'], dict(HIO_DEFAULTS), seed=20261019 + i)
                predictions['model_hio'] = reconstruct_hio(
                    batch['diffraction'], dict(HIO_DEFAULTS), seed=20261019 + i,
                    initial_object=make_model_hio_starts(
                        predictions['finetuned'], seed=20261020 + i))
                for label, prediction in predictions.items():
                    values = complex_image_metrics(prediction, batch['realspace'])
                    score = {key: float(values[key].item())
                             for key in ('complex_psnr_db', 'complex_mae')}
                    if not all(np.isfinite(value) for value in score.values()):
                        raise ValueError(f'Nonfinite metrics for {label}, sample {ids[i]}.')
                    results[label].append(score)
                np.savez_compressed(
                    recon_dir / f'{ids[i]}.npz',
                    reference_complex=batch['realspace'][0, 0].cpu().numpy(),
                    original_complex=predictions['original'][0, 0].cpu().numpy(),
                    finetuned_complex=predictions['finetuned'][0, 0].cpu().numpy(),
                    hio_complex=predictions['hio'][0, 0].cpu().numpy(),
                    model_hio_complex=predictions['model_hio'][0, 0].cpu().numpy())
                if (i + 1) % 10 == 0:
                    print(f'samples={i + 1}/100 | HIO and model+HIO complete', flush=True)
    models.clear()
    summary = {}
    for label, scores in results.items():
        passed = sum(s['complex_psnr_db'] > 20 and s['complex_mae'] < .1 for s in scores)
        summary[label] = {
            'sample_count': len(scores),
            'mean_complex_psnr_db': float(np.mean([s['complex_psnr_db'] for s in scores])),
            'mean_complex_mae': float(np.mean([s['complex_mae'] for s in scores])),
            'passed_count': passed,
            'passed_percent': 100.0 * passed / len(scores),
        }
        if label in paths:
            summary[label]['checkpoint'] = str(paths[label])
    summary['metric_protocol'] = METRIC_PROTOCOL
    summary['hio_parameters'] = dict(HIO_DEFAULTS)
    summary['hio_method'] = args.hio_method
    summary['model_initialized_hio'] = {
        'initial_checkpoint': str(paths['finetuned']),
        'initialization': 'one direct model output + one model output with fixed-seed 0.15-rad phase noise',
        'starts': 2,
        'phase_perturbation_std_rad': 0.15,
        'selection': 'lowest measured-modulus residual; reference metrics are not used',
        'same_hio_schedule_and_fixed_support': True,
    }
    summary['paired_comparison'] = {
        'mean_psnr_change_db': summary['finetuned']['mean_complex_psnr_db']
                               - summary['original']['mean_complex_psnr_db'],
        'mean_mae_change': summary['finetuned']['mean_complex_mae']
                           - summary['original']['mean_complex_mae'],
        'higher_psnr_count': sum(n['complex_psnr_db'] > o['complex_psnr_db']
                                 for o, n in zip(results['original'], results['finetuned'])),
        'lower_mae_count': sum(n['complex_mae'] < o['complex_mae']
                               for o, n in zip(results['original'], results['finetuned'])),
    }
    summary['model_hio_vs_finetuned'] = {
        'mean_psnr_change_db': summary['model_hio']['mean_complex_psnr_db']
                               - summary['finetuned']['mean_complex_psnr_db'],
        'mean_mae_change': summary['model_hio']['mean_complex_mae']
                           - summary['finetuned']['mean_complex_mae'],
        'higher_psnr_count': sum(n['complex_psnr_db'] > m['complex_psnr_db']
                                 for m, n in zip(results['finetuned'], results['model_hio'])),
        'lower_mae_count': sum(n['complex_mae'] < m['complex_mae']
                               for m, n in zip(results['finetuned'], results['model_hio'])),
    }
    summary['model_hio_vs_random_hio'] = {
        'mean_psnr_change_db': summary['model_hio']['mean_complex_psnr_db']
                               - summary['hio']['mean_complex_psnr_db'],
        'mean_mae_change': summary['model_hio']['mean_complex_mae']
                           - summary['hio']['mean_complex_mae'],
        'higher_psnr_count': sum(n['complex_psnr_db'] > h['complex_psnr_db']
                                 for h, n in zip(results['hio'], results['model_hio'])),
        'lower_mae_count': sum(n['complex_mae'] < h['complex_mae']
                               for h, n in zip(results['hio'], results['model_hio'])),
    }
    rows = [{'sample_id': sample_id,
             **{f'{label}_{key}': score[key]
             for label in results for key in ('complex_psnr_db', 'complex_mae')
                for score in [results[label][i]]}}
            for i, sample_id in enumerate(ids)]
    with (root / 'per_sample.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)
                                       + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Results: {root}', flush=True)


if __name__ == '__main__':
    main()
