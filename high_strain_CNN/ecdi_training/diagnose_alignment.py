"""Audit translation sensitivity. Uses ground truth for scoring, never training.

Reports legacy COM alignment and an additional bounded rigid translation under
the same RMS scaling. This is an alternative evaluation, not a model improvement.
Only one translation and one global phase per image are allowed. No deformation,
filtering, pixelwise phase correction, or amplitude fitting is performed.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import csv
import json
import logging
from pathlib import Path
import numpy as np


def translation_search(prediction, target, radius=4.0):
    p = np.asarray(prediction, dtype=np.complex128)
    t = np.asarray(target, dtype=np.complex128)
    fp, ft = np.fft.fft2(p), np.fft.fft2(t)
    cross = fp * np.conj(ft)
    fy, fx = np.fft.fftfreq(p.shape[0]), np.fft.fftfreq(p.shape[1])
    size = p.size
    power_p, power_t = float(np.vdot(p, p).real), float(np.vdot(t, t).real)
    def scores(ys, xs):
        ey = np.exp(-2j*np.pi*np.outer(ys, fy))
        ex = np.exp(-2j*np.pi*np.outer(xs, fx))
        return (ey @ cross @ ex.T)/size
    zero = complex(cross.sum()/size)
    best_corr, best_y, best_x = zero, 0.0, 0.0
    for step, reach in [(0.5, radius), (0.1, .5), (.02, .1), (.005, .02)]:
        ys = np.unique(np.clip(best_y + np.arange(-reach, reach+step*.25, step), -radius, radius))
        xs = np.unique(np.clip(best_x + np.arange(-reach, reach+step*.25, step), -radius, radius))
        c = scores(ys, xs)
        iy, ix = np.unravel_index(np.abs(c).argmax(), c.shape)
        if abs(c[iy, ix]) > abs(best_corr):
            best_corr, best_y, best_x = c[iy, ix], float(ys[iy]), float(xs[ix])
    ramp = np.exp(-2j*np.pi*(fy[:, None]*best_y + fx[None, :]*best_x))
    aligned = np.fft.ifft2(fp*ramp)*np.exp(-1j*np.angle(best_corr))
    return aligned, {
        'phase_only_nrmse': float(np.sqrt(max(0.0, power_p+power_t-2*abs(zero))/power_t)),
        'translation_nrmse': float(np.linalg.norm(aligned-t)/np.linalg.norm(t)),
        'shift_y_px': best_y, 'shift_x_px': best_x,
        'search_boundary_hit': int(abs(best_y) >= radius-.0051 or abs(best_x) >= radius-.0051),
        'fourier_modulus_change_rel': float(np.linalg.norm(np.abs(np.fft.fft2(aligned))-np.abs(fp))/np.linalg.norm(fp)),
    }


def score_pair(pred, truth, radius):
    from .evaluate import aligned_objects_2d
    p, t = aligned_objects_2d(pred, truth)
    p, t = p.detach().cpu().numpy(), t.detach().cpu().numpy()
    rows = []
    for i in range(len(p)):
        original, target = p[i, 0], t[i, 0]
        corrected, row = translation_search(original, target, radius)
        row['com_nrmse'] = float(np.linalg.norm(original-target)/np.linalg.norm(target))
        for name, value in [('com', original), ('translation', corrected)]:
            # Preserve the existing amplitude metric's separate max normalization.
            amp, amp_t = abs(value)/abs(value).max(), abs(target)/abs(target).max()
            row[name+'_amp_rmse'] = float(np.sqrt(np.mean((amp-amp_t)**2)))
            row[name+'_amp_rel_l1'] = float(np.sum(abs(amp-amp_t))/np.sum(amp_t))
        rows.append(row)
    return rows


def audit_saved(root, radius):
    import torch
    rows=[]
    for path in sorted(root.glob('sample_*.npz')):
        with np.load(path, allow_pickle=False) as sample:
            truth=torch.from_numpy(sample['target'].copy())[None,None]
            for stage in ['initial','refined']:
                if stage in sample:
                    row=score_pair(torch.from_numpy(sample[stage].copy())[None,None],truth,radius)[0]
                    rows.append({'sample':int(sample['sample']), 'material':str(sample['material']), 'stage':stage, **row})
    if not rows: raise FileNotFoundError(f'No sample NPZ files in {root}')
    return rows


def audit_checkpoint(args):
    import torch
    from .train import load_checkpoint, make_model, build_loader, verify_data_identity
    from .data import build_dataset, prepare_batch
    from pytorch_autophasenn.reconstruction import realspace_from_modulus_phase
    checkpoint=load_checkpoint(args.checkpoint)
    config=json.loads(json.dumps(checkpoint['config']))
    config['root']=str(Path(args.data_dir).expanduser().resolve())
    dataset=build_dataset(config,args.split,0,include_ideal=True)
    verify_data_identity(checkpoint,dataset.manifest)
    device=torch.device(args.device)
    model=make_model(checkpoint['stage'],config,checkpoint=checkpoint).to(device).eval()
    batches=build_loader(dataset,args.batch_size,args.num_workers,False,device)
    rows=[]
    with torch.no_grad():
        for number,raw in enumerate(batches,1):
            batch=prepare_batch(raw,config,device)
            if checkpoint['stage']=='phase':
                fields={'initial':realspace_from_modulus_phase(batch['diffraction'],model(batch['input']),spatial_dims=2)}
            else:
                result=model(batch['input'],batch['diffraction'])
                fields={'initial':result.initial_object,'refined':result.refined_object}
            for stage,field in fields.items():
                for i,row in enumerate(score_pair(field,batch['realspace'],args.search_radius)):
                    rows.append({'sample':int(batch['index'][i]),'material':batch['material'][i],'stage':stage,**row})
            logging.info('batch=%d/%d | samples=%d/%d',number,len(batches),len({r['sample'] for r in rows}),len(dataset))
    return rows, {'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_epoch':checkpoint['epoch'],'data':dataset.manifest}


def write_report(rows, output, provenance, radius):
    output.mkdir(parents=True,exist_ok=False)
    with (output/'per_sample.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups={}
    for material in ['ALL']+sorted({r['material'] for r in rows}):
        for stage in sorted({r['stage'] for r in rows}):
            selected=[r for r in rows if r['stage']==stage and (material=='ALL' or r['material']==material)]
            if not selected: continue
            groups[material+'/'+stage]={'n':len(selected),**{k:float(np.mean([r[k] for r in selected])) for k in rows[0] if k not in ['sample','material','stage']}}
    report={'provenance':provenance,'search_radius_px':radius,'groups':groups,'notes':[
        'Ground-truth-based rigid translation and global phase are used only to score reconstruction.',
        'No training or model modification. Scores must not be presented as improved inference.',
        'COM and translation scores use different evaluation definitions; retain both.',
        'A bounded coarse-to-fine search is used; it is not proof of a global optimum.',
        'The conjugate twin branch is inherited from legacy alignment, not searched again.',
        'Boundary hits mean the specified search may be too narrow; inspect before interpreting.',
        'Saved samples are a small convenience subset, not an estimate of full validation performance.',
    ]}
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for key,values in groups.items():
        if key.startswith('ALL/'):
            logging.info('%s | n=%d | COM=%.6f | translation=%.6f | boundary_fraction=%.3f',key,values['n'],values['com_nrmse'],values['translation_nrmse'],values['search_boundary_hit'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint');source.add_argument('--samples-dir',type=Path)
    parser.add_argument('--data-dir',default='/data_ssd/oyys/eCDI')
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--batch-size',type=int,default=12)
    parser.add_argument('--num-workers',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--search-radius',type=float,default=4.)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.batch_size<1 or args.num_workers<0 or not 0<args.search_radius<=8:
        parser.error('Require batch>=1, workers>=0, 0<search-radius<=8')
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new name to preserve previous results.')
    import torch
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    if args.samples_dir:
        rows=audit_saved(args.samples_dir,args.search_radius);provenance={'saved_samples':str(args.samples_dir.resolve())}
    else:
        rows,provenance=audit_checkpoint(args)
    write_report(rows,args.output_dir,provenance,args.search_radius)
    logging.info('Output=%s',args.output_dir)


if __name__=='__main__':
    main()
