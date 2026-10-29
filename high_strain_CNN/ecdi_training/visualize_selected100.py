"""Create per-sample metric bars and representative reconstructions for 100 samples.

This is post-processing only: it does not run inference, HIO, or modify benchmark
records. Run from high_strain_CNN with ``python -m ecdi_training.visualize_selected100``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_SAMPLES = Path('/data_ssd/oyys/eCDI/test_data/ecdi_100/ecdi_100_samples.h5')


def read_metrics(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise ValueError(f'Expected 100 metric rows, found {len(rows)} in {path}')
    if not rows:
        raise ValueError(f'No metric rows in {path}')
    if 'finetuned_complex_psnr_db' in rows[0]:
        mapping = {
            'finetuned_complex_psnr_db': 'system_PSNR_dB',
            'finetuned_complex_mae': 'system_MAE',
            'hio_complex_psnr_db': 'traditional_PSNR_dB',
            'hio_complex_mae': 'traditional_MAE',
            'model_hio_complex_psnr_db': 'model_hio_PSNR_dB',
            'model_hio_complex_mae': 'model_hio_MAE',
        }
        for row in rows:
            for source, target in mapping.items():
                if source in row:
                    row[target] = row[source]
    required = {'sample_id', 'system_PSNR_dB', 'system_MAE',
                'traditional_PSNR_dB', 'traditional_MAE'}
    if not required.issubset(rows[0]):
        raise ValueError(f'Metrics CSV must contain columns: {sorted(required)}')
    ids = [row['sample_id'] for row in rows]
    if len(set(ids)) != 100:
        raise ValueError('Metrics CSV has duplicate sample IDs.')
    for row in rows:
        for field in {key for key in row if key != 'sample_id'}:
            row[field] = float(row[field])
            if not np.isfinite(row[field]):
                raise ValueError(f'Nonfinite {field} for sample {row["sample_id"]}')
    return rows


def aligned_for_display(prediction: np.ndarray, target: np.ndarray):
    """Apply the same translation/twin/global-phase/RMS alignment as scoring."""
    import torch
    from .periodic_metrics import align_periodic_fields

    pred = torch.from_numpy(np.asarray(prediction)).to(torch.complex64)[None, None]
    truth = torch.from_numpy(np.asarray(target)).to(torch.complex64)[None, None]
    pred, truth = align_periodic_fields(pred, truth)
    shared_peak = truth.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return (pred[0, 0].cpu().numpy() / shared_peak.item(),
            truth[0, 0].cpu().numpy() / shared_peak.item())


def make_metric_figure(rows, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})
    psnr = np.array([r['system_PSNR_dB'] for r in rows])
    mae = np.array([r['system_MAE'] for r in rows])
    # Fixed, readable bins make the 100-sample distributions visually compact.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), constrained_layout=True)
    style = {'color': '#2878B5', 'edgecolor': '#2878B5', 'linewidth': .25}
    axes[0].hist(psnr, bins=20, **style)
    axes[0].axvline(20, color='#C51B29', linestyle='--', linewidth=1.25,
                    label='20 dB threshold')
    axes[0].set_xlabel('Complex PSNR (dB)')
    axes[0].set_ylabel('Count')
    axes[0].text(.04, .95, 'a', transform=axes[0].transAxes, va='top',
                 fontweight='bold', fontsize=13)
    axes[1].hist(mae, bins=20, **style)
    axes[1].axvline(.1, color='#C51B29', linestyle='--', linewidth=1.25,
                    label='0.1 threshold')
    axes[1].set_xlabel('Complex MAE')
    axes[1].set_ylabel('Count')
    axes[1].text(.04, .95, 'b', transform=axes[1].transAxes, va='top',
                 fontweight='bold', fontsize=13)
    for ax in axes:
        ax.tick_params(direction='out', length=3, width=.8, labelsize=9)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.legend(frameon=False, fontsize=7, loc='best')
    fig.savefig(output, bbox_inches='tight')
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)


def make_sample_figure(samples, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    has_model_hio = all('model_hio' in sample for sample in samples)
    field_names = ['Reference', 'Neural model', 'HIO'] + (['Model + HIO'] if has_model_hio else [])
    nfields = len(field_names)
    fig, axes = plt.subplots(3, 2 * nfields, figsize=(14.5, 6.3), squeeze=False)
    fig.subplots_adjust(left=.025, right=.91, bottom=.035, top=.88,
                        wspace=.11, hspace=.16)
    peak = max(max(float(np.abs(s['truth']).max()), float(np.abs(s['pred']).max()),
                   float(np.abs(s['hio']).max()),
                   float(np.abs(s['model_hio']).max()) if has_model_hio else 0.0)
               for s in samples)
    phase_cmap = plt.get_cmap('twilight').copy()
    phase_cmap.set_bad('white')
    for row_index, sample in enumerate(samples):
        pred, truth = aligned_for_display(sample['pred'], sample['truth'])
        hio, _ = aligned_for_display(sample['hio'], sample['truth'])
        fields = [truth, pred, hio]
        if has_model_hio:
            model_hio, _ = aligned_for_display(sample['model_hio'], sample['truth'])
            fields.append(model_hio)
        reference_amplitude = np.abs(truth)
        phase_support = reference_amplitude >= 0.05 * reference_amplitude.max()
        for col, field in enumerate(fields):
            axes[row_index, col].imshow(np.abs(field), cmap='magma', vmin=0, vmax=peak)
            phase = np.ma.masked_where(~phase_support, np.angle(field))
            axes[row_index, col + nfields].imshow(phase, cmap=phase_cmap,
                                                   vmin=-np.pi, vmax=np.pi)
        for ax in axes[row_index]:
            ax.axis('off')
        metric_columns = [
            (1, 'system_PSNR_dB', 'system_MAE'),
            (2, 'traditional_PSNR_dB', 'traditional_MAE'),
        ]
        if has_model_hio:
            metric_columns.append((3, 'model_hio_PSNR_dB', 'model_hio_MAE'))
        for col, psnr_key, mae_key in metric_columns:
            if psnr_key in sample['metrics']:
                axes[row_index, col].text(
                    .035, .965,
                    f"PSNR {sample['metrics'][psnr_key]:.2f} dB\n"
                    f"MAE {sample['metrics'][mae_key]:.4f}",
                    transform=axes[row_index, col].transAxes, ha='left', va='top',
                    color='white', fontsize=7.5, linespacing=1.2,
                    bbox={'facecolor': 'black', 'alpha': .68, 'edgecolor': 'none', 'pad': 2.5})
    short_names = ['Truth', 'Model', 'HIO'] + (['Model + HIO'] if has_model_hio else [])
    labels = [f'{name}\nAmplitude' for name in short_names]
    labels.extend(f'{name}\nPhase' for name in short_names)
    for col, label in enumerate(labels):
        axes[0, col].set_title(label, fontsize=8.5, pad=4, linespacing=1.05)
    fig.colorbar(axes[0, 1].images[0], ax=axes[:, :nfields], shrink=.78,
                 pad=.015, label='Amplitude')
    fig.colorbar(axes[0, nfields + 1].images[0], ax=axes[:, nfields:], shrink=.78,
                 pad=.015, label='Phase (radians)')
    fig.suptitle('Representative reconstructions', fontsize=12, y=.97)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path,
                        help='Completed benchmark output containing per_sample.csv and reconstructions/.')
    parser.add_argument('--samples-h5', type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument('--output-dir', type=Path,
                        help='Defaults to RUN_DIR/figures; benchmark outputs remain unchanged.')
    args = parser.parse_args()

    try:
        import h5py
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f'Missing plotting/data dependency: {exc}. Install matplotlib and h5py in the mamba environment.') from exc

    run_dir = args.run_dir.expanduser().resolve()
    h5_path = args.samples_h5.expanduser().resolve()
    metrics_path = run_dir / 'per_sample.csv'
    recon_dir = run_dir / 'reconstructions'
    for required_path in (metrics_path, recon_dir, h5_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    rows = read_metrics(metrics_path)
    out_dir = (args.output_dir or run_dir / 'figures').expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    make_metric_figure(rows, out_dir / 'per_sample_metrics.png')

    # Select best, median, and worst by model PSNR, with deterministic tie breaks.
    ranked = sorted(rows, key=lambda r: (r['system_PSNR_dB'], r['sample_id']))
    picks = [('worst', ranked[0]), ('median', ranked[len(ranked) // 2]), ('best', ranked[-1])]
    selected = []
    with h5py.File(h5_path, 'r') as data:
        ids = data['display_id'].asstr()[:].tolist()
        id_to_index = {sample_id: i for i, sample_id in enumerate(ids)}
        for label, row in picks:
            sample_id = row['sample_id']
            if sample_id not in id_to_index:
                raise ValueError(f'Sample {sample_id} from metrics is absent from the HDF5 bundle.')
            npz_path = recon_dir / f'{sample_id}.npz'
            if not npz_path.exists():
                raise FileNotFoundError(npz_path)
            with np.load(npz_path) as saved:
                if 'finetuned_complex' in saved:
                    pred = saved['finetuned_complex']
                    hio = saved['hio_complex']
                    model_hio = saved['model_hio_complex']
                    truth = saved['reference_complex']
                else:
                    pred = saved['system_complex']
                    hio = saved['traditional_complex']
                    model_hio = None
                    truth = None
            i = id_to_index[sample_id]
            if truth is None:
                truth = data['exit_wave_real'][i] + 1j * data['exit_wave_imag'][i]
            item = {'category': label, 'sample_id': sample_id,
                    'metrics': row, 'pred': pred, 'hio': hio, 'truth': truth}
            if model_hio is not None:
                item['model_hio'] = model_hio
            selected.append(item)
    make_sample_figure(selected, out_dir / 'representative_samples.png')

    manifest = {'sample_count': 100,
                'metric_figure': 'per_sample_metrics.png',
                'sample_figure': 'representative_samples.png',
                'complex_metric_definition': 'Values are the existing per_sample.csv full-field complex PSNR and MAE; figures do not recompute or change scores.',
                'phase_display_mask': 'Show phase only where reference amplitude is at least 5% of its peak; masked pixels are white. This affects display only, not metrics.',
                'representative_sample_rule': 'Worst, median, and best sample by neural-model PSNR among the 100 rows; deterministic ID tie break.',
                'representative_samples': [
                    {'category': item['category'], 'sample_id': item['sample_id'],
                     'system_PSNR_dB': item['metrics']['system_PSNR_dB'],
                     'system_MAE': item['metrics']['system_MAE']}
                    for item in selected]}
    (out_dir / 'figure_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Figures saved to: {out_dir}')
    print('  per_sample_metrics.png')
    print('  representative_samples.png (worst, median, and best by model PSNR)')


if __name__ == '__main__':
    main()
