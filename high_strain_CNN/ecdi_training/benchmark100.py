"""Frozen 100-sample comparison: cascade, random-start HIO, and cascade-start HIO.

Run from high_strain_CNN: python -u -m ecdi_training.benchmark100
Truth is used ONLY for evaluation. Both HIO paths use measured modulus and
measurement-derived shrinkwrap supports. One starts randomly; the other starts
from the neural reconstruction. Neither HIO path uses truth.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = Path('/data_ssd/oyys/eCDI/test_data/ecdi_100')
SAMPLE_FILES = {
    'h5': 'ecdi_100_samples.h5',
    'csv': 'ecdi_100_samples.csv',
    'provenance': 'ecdi_100_provenance.json',
    'checksums': 'SHA256SUMS.txt',
}
LOG = logging.getLogger(__name__)
HIO_DEFAULTS = {
    'beta': .9, 'hio_iterations': 1500, 'er_iterations': 100, 'restarts': 4,
    'initial_autocorrelation_threshold': .04, 'shrinkwrap_every': 20,
    'shrinkwrap_until': 1200, 'shrinkwrap_threshold': .20,
    'sigma_start_px': 3., 'sigma_min_px': 1.5, 'sigma_decay': .99,
}
HIO_PARAMETER_SOURCE = (
    'Marchesini et al. (2003): beta=0.9, initial autocorrelation contour=4%, '
    'shrinkwrap every 20 iterations, threshold=20% of blurred amplitude maximum, '
    'Gaussian sigma=3 px reduced by 1% per update to 1.5 px. Project-fixed settings '
    'not specified by that paper: 1500 HIO iterations, 100 terminal ER iterations, '
    'four random starts, and stopping support updates at iteration 1200.'
)
METRIC_DESCRIPTION = (
    'Full 512x512 COMPLEX field. Original evaluation alignment: amplitude COM, '
    'MAE-selected direct/conjugate-twin, global phase fitted on the reference '
    'amplitude>=0.3 support. Each field is whole-image RMS normalized; BOTH are '
    'then divided by the same aligned-reference peak modulus. '
    'MAE=mean(abs(pred-reference)); MSE=mean(abs(pred-reference)**2); '
    'PSNR=-10*log10(MSE), peak modulus=1. No scoring crop/mask, no clipping, '
    'and no real/imaginary-channel averaging. Local phase errors count.'
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str), encoding='utf-8')


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def validate_hio(config):
    if set(config) != set(HIO_DEFAULTS):
        raise ValueError('HIO configuration must contain exactly the documented keys.')
    for key in ('hio_iterations', 'er_iterations', 'restarts', 'shrinkwrap_every', 'shrinkwrap_until'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if config['shrinkwrap_until'] > config['hio_iterations']:
        raise ValueError('Shrinkwrap must stop no later than the last HIO iteration.')
    for key in ('beta', 'initial_autocorrelation_threshold', 'shrinkwrap_threshold', 'sigma_decay'):
        if not math.isfinite(config[key]) or not 0 < config[key] < 1:
            raise ValueError(f'{key} must be finite and strictly between 0 and 1.')
    if not 0 < config['sigma_min_px'] <= config['sigma_start_px']:
        raise ValueError('Require 0 < sigma_min_px <= sigma_start_px.')
    if not all(math.isfinite(config[k]) for k in ('sigma_min_px', 'sigma_start_px')):
        raise ValueError('Gaussian widths must be finite.')


def prepare_measurement(counts, config, device):
    """Same intensity/log/min-max operations as data.prepare_batch, without truth."""
    import torch
    measured = torch.from_numpy(counts).to(device=device, dtype=torch.float32)[None, None]
    if not bool(torch.isfinite(measured).all() & (measured >= 0).all()):
        raise ValueError('Invalid input counts.')
    eps = torch.finfo(measured.dtype).eps
    modulus = measured.sqrt()
    if not bool(modulus.amax() > 0):
        raise ValueError('All-zero diffraction input.')
    modulus = modulus / modulus.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    image = torch.log1p(measured) if config.get('input_log_data', True) else measured
    minimum = image.amin(dim=(-2, -1), keepdim=True)
    spread = image.amax(dim=(-2, -1), keepdim=True) - minimum
    if not bool((spread > 0).all()):
        raise ValueError('Constant diffraction input.')
    image = (image - minimum) / spread.clamp_min(eps)
    return image, modulus


def fftc(field):
    import torch
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1))


def ifftc(field):
    import torch
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1))


def reconstruct_hio(modulus, config, seed, initial_field=None):
    """Complex HIO followed by ER, with four default independent random starts.

    HIO: inside support use modulus-projected field; outside use old-beta*projected.
    ER: modulus projection followed by zeroing outside support. No positivity.
    Choose the restart with the smallest observed-modulus residual, NEVER PSNR.
    Shrinkwrap settings follow Marchesini et al. (2003). Iteration and restart
    counts are fixed project comparison settings recorded in run_config.json.
    """
    import torch
    k = config['restarts'] if initial_field is None else 1
    h, w = modulus.shape[-2:]
    m = modulus[0, 0]
    autocorrelation = ifftc(m.square()).abs()
    support0 = autocorrelation >= config['initial_autocorrelation_threshold'] * autocorrelation.amax()
    if not bool(support0.any()):
        raise RuntimeError('Empty autocorrelation support.')
    support = support0[None].expand(k, h, w).clone()
    if initial_field is None:
        generator = torch.Generator(device=modulus.device).manual_seed(seed)
        phases = (torch.rand((k, h, w), device=modulus.device, generator=generator) * 2 - 1) * math.pi
        fields = ifftc(torch.polar(m.expand(k, h, w), phases))
    else:
        fields = torch.as_tensor(initial_field, device=modulus.device)
        if not fields.is_complex():
            fields = fields.to(torch.complex64)
        fields = fields.reshape(1, h, w).clone()
    fy = torch.fft.fftfreq(h, device=modulus.device)
    fx = torch.fft.fftfreq(w, device=modulus.device)
    freq_squared = fy[:, None].square() + fx[None, :].square()
    sigma = config['sigma_start_px']
    gaussian = torch.exp(-2 * math.pi**2 * sigma**2 * freq_squared)
    last_progress = time.monotonic()
    init_label = 'model reconstruction' if initial_field is not None else f'random (seed={seed})'
    LOG.info('HIO starts=%d | initialization=%s | iterations=%d+%d ER',
             k, init_label, config['hio_iterations'], config['er_iterations'])
    for step in range(1, config['hio_iterations'] + 1):
        spectrum = fftc(fields)
        projected = ifftc(torch.polar(m.expand_as(spectrum.real), torch.angle(spectrum)))
        fields = torch.where(support, projected, fields - config['beta'] * projected)
        if step % config['shrinkwrap_every'] == 0 and step <= config['shrinkwrap_until']:
            blurred = torch.fft.ifft2(torch.fft.fft2(fields.abs()) * gaussian).real.clamp_min(0)
            maximum = blurred.amax(dim=(-2, -1), keepdim=True)
            support = blurred >= config['shrinkwrap_threshold'] * maximum
            sigma = max(config['sigma_min_px'], sigma * config['sigma_decay'])
            gaussian = torch.exp(-2 * math.pi**2 * sigma**2 * freq_squared)
        if step % 100 == 0 and time.monotonic() - last_progress >= 20:
            LOG.info('HIO iteration=%d/%d', step, config['hio_iterations'])
            last_progress = time.monotonic()
    for _ in range(config['er_iterations']):
        spectrum = fftc(fields)
        projected = ifftc(torch.polar(m.expand_as(spectrum.real), torch.angle(spectrum)))
        fields = torch.where(support, projected, torch.zeros_like(projected))
    residuals = ((fftc(fields).abs() - m).square().sum(dim=(-2, -1)) / m.square().sum()).sqrt()
    if not bool(torch.isfinite(fields).all() & torch.isfinite(residuals).all()):
        raise RuntimeError('Nonfinite HIO reconstruction.')
    best = int(residuals.argmin())
    return fields[best][None, None]


def timed_call(action, device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start_utc = stamp()
    start = time.perf_counter()
    result = action()  # System reconstruction and final complex-array transfer to CPU.
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    return result, {'start_utc': start_utc, 'end_utc': stamp(), 'seconds': seconds}


def preflight(folder):
    import h5py
    from .select_original_train100_complex import digest
    names = {SAMPLE_FILES['h5'], SAMPLE_FILES['csv'], SAMPLE_FILES['provenance']}
    checksums = {}
    for line in (folder / SAMPLE_FILES['checksums']).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        value, name = line.split(None, 1)
        name = name.strip()
        if name not in names or name in checksums:
            raise ValueError('Unexpected or duplicate checksum entry.')
        checksums[name] = value
    if set(checksums) != names:
        raise ValueError('Incomplete sample-bundle checksums.')
    for name, expected in checksums.items():
        if digest(folder / name) != expected:
            raise ValueError(f'Sample checksum mismatch: {name}')
    manifest = json.loads((folder / SAMPLE_FILES['provenance']).read_text(encoding='utf-8'))
    if manifest.get('num_samples') != 100:
        raise ValueError('Exactly 100 frozen samples are required.')
    if manifest.get('confirmed_precision_thresholds') != {'complex_psnr_db_strict_gt': 20., 'complex_mae_strict_lt': .1}:
        raise ValueError('Sample bundle does not use the confirmed complex precision thresholds.')
    with h5py.File(folder / SAMPLE_FILES['h5'], 'r') as f:
        for name in ('observed_counts', 'exit_wave_real', 'exit_wave_imag'):
            if f[name].shape != (100, 512, 512) or f[name].dtype != np.dtype('float32'):
                raise ValueError(f'Invalid data shape/dtype: {name}')
        ids = f['display_id'].asstr()[:].tolist()
        source_ids = f['source_index'][:].tolist()
        if ids != [f'{i:04d}' for i in range(1, 101)] or len(set(source_ids)) != 100:
            raise ValueError('Invalid or duplicate sample IDs.')
        if not f['valid'][:].all():
            raise ValueError('Invalid sample flags.')
        if ids != [r['display_id'] for r in manifest['samples']] or source_ids != [r['source_index'] for r in manifest['samples']]:
            raise ValueError('Array identities do not match the frozen selection.')
        for i in range(100):
            counts = f['observed_counts'][i]
            truth_real = f['exit_wave_real'][i]
            truth_imag = f['exit_wave_imag'][i]
            for name, array in (('observed_counts', counts),
                                ('exit_wave_real', truth_real),
                                ('exit_wave_imag', truth_imag)):
                if not np.isfinite(array).all() or (name == 'observed_counts' and (array < 0).any()):
                    raise ValueError(f'Invalid values in sample {ids[i]}: {name}')
            if counts.max() <= 0:
                raise ValueError(f'Zero input for {ids[i]}')
            if not np.any(truth_real) and not np.any(truth_imag):
                raise ValueError(f'Zero reference for {ids[i]}')
    return manifest, checksums


def environment(device):
    import torch
    result = {'platform': platform.platform(), 'cpu': platform.processor(), 'python': sys.version,
              'torch': str(torch.__version__), 'torch_cuda_runtime': torch.version.cuda,
              'execution_device': str(device), 'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES')}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, gpu_total_bytes=props.total_memory)
    try:
        cmd = ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.used', '--format=csv,noheader']
        result['nvidia_smi'] = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        result['nvidia_smi'] = None
    try:
        import psutil
        result['ram_total_bytes'] = psutil.virtual_memory().total
    except ImportError:
        result['ram_total_bytes'] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-dir', type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument('--checkpoint', default='', help='Defaults to the frozen original checkpoint in the sample manifest.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=20261019)
    args = parser.parse_args()
    import h5py
    import torch
    from .select_original_train100_complex import complex_image_metrics, digest
    from .train import load_checkpoint, make_model
    if not 0 <= args.seed < 2**63 - 100000:
        parser.error('Seed must be a nonnegative signed 64-bit integer with room for sample offsets.')
    hio = copy.deepcopy(HIO_DEFAULTS)
    validate_hio(hio)
    folder = args.samples_dir.expanduser().resolve()
    manifest, dataset_sha256 = preflight(folder)
    checkpoint_path = Path(args.checkpoint or manifest['checkpoint']).expanduser().resolve()
    if digest(checkpoint_path) != manifest['checkpoint_sha256']:
        raise ValueError('Use the exact original checkpoint frozen with these samples.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    root = args.output_dir or PROJECT / 'artifacts/evaluations' / f'ecdi_100_benchmark_{datetime.now():%Y%m%d_%H%M%S}'
    if root.exists():
        raise FileExistsError('Use a fresh output directory; previous records are preserved.')
    root.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(root / 'run.log', encoding='utf-8')])
    rows = []
    try:
        torch.manual_seed(args.seed)
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get('stage') != 'refiner':
            raise ValueError('A complete original phase + refiner checkpoint is required.')
        config = copy.deepcopy(checkpoint['config'])
        if config.get('shape') != [512, 512] or config.get('diffraction_kind') != 'intensity':
            raise ValueError('Expected the original 512x512 intensity-input model.')
        from . import evaluate as evaluation_module
        from . import select_original_train100_complex as metric_module
        from . import data as data_module
        from . import train as train_module
        import pytorch_autophasenn.reconstruction as reconstruction_module
        import pytorch_autophasenn.model as phase_module
        import pytorch_autophasenn.momamba_refiner as refiner_module
        source_paths = {
            'benchmark': Path(__file__),
            'evaluation_metrics': Path(evaluation_module.__file__),
            'metric_definition': Path(metric_module.__file__),
            'data_loading': Path(data_module.__file__),
            'model_loading': Path(train_module.__file__),
            'physics_reconstruction': Path(reconstruction_module.__file__),
            'phase_model': Path(phase_module.__file__),
            'refiner_model': Path(refiner_module.__file__),
        }
        record = {'purpose': 'Internal exploratory comparison only; not a formal acceptance test.',
                  'sample_count': 100, 'sample_bundle': str(folder), 'checkpoint': str(checkpoint_path),
                  'dataset_sha256': {SAMPLE_FILES['h5']: dataset_sha256[SAMPLE_FILES['h5']],
                                     SAMPLE_FILES['csv']: dataset_sha256[SAMPLE_FILES['csv']]},
                  'checkpoint_sha256': manifest['checkpoint_sha256'], 'checkpoint_epoch': checkpoint['epoch'],
                  'metric_protocol': METRIC_DESCRIPTION,
                  'precision_thresholds': {'psnr_db_strict_gt': 20., 'mae_strict_lt': .1, 'required_count': 95},
                  'system_time_requirement': 'Every sample < 300 seconds; 100/100 required.',
                  'timing_scope': 'System timing starts after the sample has been read, transferred CPU-to-device, and preprocessed. It includes model inference and copying the final complex array to CPU memory. It excludes model loading, file reading, transfer, preprocessing, truth alignment, scoring and disk writes.',
                  'traditional_algorithm': 'Complex HIO + terminal ER, adaptive shrinkwrap; no positivity constraint. Baseline uses four random starts and selects by measured-modulus residual. Additional run uses the neural reconstruction as its single initial field.',
                  'model_initialized_hio': 'Starts from the neural reconstruction; then applies the same measured-modulus projection, HIO update, adaptive shrinkwrap schedule, and terminal ER as the traditional baseline. Ground truth is not used for initialization or restart selection.',
                  'hio_parameters': hio, 'hio_parameter_source_and_deviations': HIO_PARAMETER_SOURCE,
                  'seed_rule': 'global seed + zero-based sample display order', 'seed': args.seed,
                  'references': ['https://doi.org/10.1364/AO.21.002758', 'https://arxiv.org/abs/physics/0306174'],
                  'source_code_sha256': {role: digest(path) for role, path in source_paths.items()},
                  'environment': environment(device)}
        dump(root / 'run_config.json', record)
        config_hash = digest(root / 'run_config.json')
        output_dir = root / 'reconstructions'
        output_dir.mkdir()
        model = make_model('refiner', config, checkpoint=checkpoint).to(device).eval()
        LOG.info('Input count=100 | original checkpoint epoch=%s | batch_size=1', checkpoint['epoch'])
        LOG.info('HIO fixed parameters=%s', hio)
        LOG.info('Runs per sample: neural model, random-start HIO, and model-initialized HIO.')
        LOG.info('PSNR > 20dB AND MAE < 0.1; system reconstruction time < 300s per sample.')
        csv_path = root / 'per_sample.partial.csv'
        with h5py.File(folder / SAMPLE_FILES['h5'], 'r') as f, csv_path.open('w', newline='', encoding='utf-8-sig') as handle, torch.no_grad():
            writer = None
            for i in range(100):
                display_id = f['display_id'].asstr()[i]
                counts = f['observed_counts'][i]
                # The plan excludes sample loading, CPU-to-GPU transfer and
                # preprocessing. Prepare once, then time the system reconstruction.
                image, modulus = prepare_measurement(counts, config, device)

                def system_action():
                    return model(image, modulus).refined_object[0, 0].detach().cpu().numpy().copy()

                def hio_action():
                    field = reconstruct_hio(modulus, hio, args.seed + i)
                    return field[0, 0].detach().cpu().numpy().copy()

                def model_initialized_hio_action():
                    field = reconstruct_hio(modulus, hio, args.seed + i,
                                            initial_field=system)
                    return field[0, 0].detach().cpu().numpy().copy()

                system, system_time = timed_call(system_action, device)
                traditional = hio_action()
                model_hio, model_hio_time = timed_call(model_initialized_hio_action, device)
                if not (np.isfinite(system).all() and np.isfinite(traditional).all()
                        and np.isfinite(model_hio).all()):
                    raise RuntimeError(f'Nonfinite output for sample {display_id}')
                # Ground truth enters ONLY here, after both reconstructions and timers finish.
                truth = torch.complex(torch.from_numpy(f['exit_wave_real'][i]),
                                      torch.from_numpy(f['exit_wave_imag'][i]))[None, None].to(device)
                predictions = [torch.from_numpy(a)[None, None].to(device)
                               for a in (system, traditional, model_hio)]
                values = [complex_image_metrics(p, truth) for p in predictions]
                psnr_s, psnr_h, psnr_mh = [float(v['complex_psnr_db'][0]) for v in values]
                mae_s, mae_h, mae_mh = [float(v['complex_mae'][0]) for v in values]
                if not all(math.isfinite(x) for x in (psnr_s, psnr_h, psnr_mh,
                                                       mae_s, mae_h, mae_mh)):
                    raise RuntimeError(f'Nonfinite metrics for sample {display_id}')
                row = {'sample_id': display_id,
                       'system_start_utc': system_time['start_utc'], 'system_end_utc': system_time['end_utc'],
                       'system_minutes': system_time['seconds'] / 60.,
                       'system_time_pass': system_time['seconds'] < 300.,
                       'system_PSNR_dB': psnr_s, 'system_MAE': mae_s,
                       'system_precision_pass': psnr_s > 20. and mae_s < .1,
                       'traditional_PSNR_dB': psnr_h, 'traditional_MAE': mae_h,
                       'model_initialized_hio_seconds': model_hio_time['seconds'],
                       'model_initialized_hio_PSNR_dB': psnr_mh,
                       'model_initialized_hio_MAE': mae_mh}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                np.savez_compressed(output_dir / f'{display_id}.npz',
                                    system_complex=system, traditional_complex=traditional,
                                    model_initialized_hio_complex=model_hio)
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                LOG.info('%s/0100 | model PSNR=%.4f MAE=%.6f time=%.3fs | '
                         'random-HIO PSNR=%.4f MAE=%.6f | model-init-HIO PSNR=%.4f MAE=%.6f time=%.3fs',
                         display_id, psnr_s, mae_s, system_time['seconds'], psnr_h, mae_h,
                         psnr_mh, mae_mh, model_hio_time['seconds'])
        if len(rows) != 100 or len({r['sample_id'] for r in rows}) != 100:
            raise RuntimeError('All 100 samples must complete; partial results cannot pass.')
        if digest(root / 'run_config.json') != config_hash:
            raise RuntimeError('Run configuration changed during evaluation.')
        csv_path.rename(root / 'per_sample.csv')
        s_psnr = float(np.mean([r['system_PSNR_dB'] for r in rows]))
        h_psnr = float(np.mean([r['traditional_PSNR_dB'] for r in rows]))
        s_mae = float(np.mean([r['system_MAE'] for r in rows]))
        h_mae = float(np.mean([r['traditional_MAE'] for r in rows]))
        mh_psnr = float(np.mean([r['model_initialized_hio_PSNR_dB'] for r in rows]))
        mh_mae = float(np.mean([r['model_initialized_hio_MAE'] for r in rows]))
        n_time = sum(r['system_time_pass'] for r in rows)
        n_acc = sum(r['system_precision_pass'] for r in rows)
        comparison_pass = s_psnr > h_psnr and s_mae < h_mae
        LOG.info('DONE | model mean PSNR=%.4f MAE=%.6f | random-HIO mean PSNR=%.4f MAE=%.6f | model-init-HIO mean PSNR=%.4f MAE=%.6f',
                 s_psnr, s_mae, h_psnr, h_mae, mh_psnr, mh_mae)
        LOG.info('Time pass=%d/100 | precision pass=%d/100 | comparison pass=%s | outputs=%s', n_time, n_acc, comparison_pass, root.resolve())
        LOG.info('SUMMARY | samples=100 | max_system_minutes=%.4f | precision_pass=%d/100 (%.1f%%) | numerical_criteria_pass=%s | internal exploratory comparison only; not a formal acceptance result',
                 max(r['system_minutes'] for r in rows), n_acc, n_acc, n_time == 100 and n_acc >= 95 and comparison_pass)
    except BaseException as error:
        LOG.exception('Stopped; original/partial records preserved. Use a new output directory for another run.')
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
