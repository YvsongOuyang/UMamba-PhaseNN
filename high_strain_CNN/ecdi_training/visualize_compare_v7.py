"""Render matched old/new reconstructions; no training or model selection.

Choose median and 90th-percentile baseline-error samples per material, plus
the two largest regressions. These descriptive test illustrations must not be
used to tune the model. Labels are used only for scoring/visual registration.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .data_v2 import build_dataset,prepare_batch
from .train import load_checkpoint,make_model,verify_data_identity
from .aligned_objective_v7 import align_objects,OBJECTIVE_VERSION
from .train_aligned_v7 import read_thickness,state_digest,file_digest


def read_scores(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r['stage']=='refined']
    result={int(r['sample']):r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate refined sample identities.')
    return result


def select_examples(old,new):
    if old.keys()!=new.keys():raise ValueError('Old/new score sample sets differ.')
    if any(old[i]['material']!=new[i]['material'] for i in old):raise ValueError('Material mismatch.')
    selected={}
    def add(i,reason):selected.setdefault(i,[]).append(reason)
    for material in sorted({r['material'] for r in old.values()}):
        ordered=sorted([i for i in old if old[i]['material']==material],key=lambda i:(float(old[i]['registered_complex_nrmse']),i))
        for fraction,name in [(.5,'baseline_median'),(.9,'baseline_90th_percentile')]:
            add(ordered[round((len(ordered)-1)*fraction)],name)
    regressions=sorted(old,key=lambda i:float(new[i]['registered_complex_nrmse'])-float(old[i]['registered_complex_nrmse']),reverse=True)
    for i in regressions[:2]:
        if float(new[i]['registered_complex_nrmse'])>float(old[i]['registered_complex_nrmse']):add(i,'largest_regression')
    return selected


def render(path,sid,material,thickness,predictions,truth,reasons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cy,cx=truth.shape[0]//2,truth.shape[1]//2
    crop=np.s_[cy-48:cy+48,cx-48:cx+48]
    reference_peak=max(float(abs(truth).max()),1e-12)
    objects=[('Target',truth)]+list(predictions.items())
    amplitude_max=max(float(abs(p).max()/reference_peak) for _,p in objects)
    error_max=max(float(abs(p-truth)[crop].max()) for _,p in objects)
    error_max=max(error_max,1e-6)
    support=abs(truth)>=.3*reference_peak
    fig,axes=plt.subplots(3,4,figsize=(16,10),layout='constrained')
    for row,(label,p) in enumerate(objects):
        nrmse=float(np.linalg.norm(p-truth)/np.linalg.norm(truth))
        amplitude=abs(p)/reference_peak
        phase_error=np.where(support,np.angle(p*np.conj(truth)),np.nan)
        panels=[(amplitude,'Full amplitude','viridis',0,amplitude_max),
                (amplitude[crop],'Central 96x96 amplitude','viridis',0,amplitude_max),
                (phase_error[crop],'Wrapped phase error (rad)','RdBu_r',-.5,.5),
                (abs(p-truth)[crop],'Complex error (RMS scale)','inferno',0,error_max)]
        for col,(array,title,cmap,lo,hi) in enumerate(panels):
            ax=axes[row,col]
            im=ax.imshow(array,cmap=cmap,vmin=lo,vmax=hi)
            ax.set_title(f'{label} | {title}',fontsize=10);ax.axis('off')
            fig.colorbar(im,ax=ax,shrink=.8)
        axes[row,0].set_ylabel(f'NRMSE={nrmse:.4f}')
        axes[row,0].set_title(f'{label} | Full amplitude\nNRMSE={nrmse:.4f}',fontsize=10)
    fig.suptitle(f'Sample {sid} | {material} | thickness {thickness:.1f} A\nSelection: {", ".join(reasons)} | same RMS convention and rigid-registration rule')
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-checkpoint',required=True)
    parser.add_argument('--new-checkpoint',required=True)
    parser.add_argument('--old-evaluation-dir',type=Path,default=Path('artifacts/evaluations/refiner_v3_registered_test'))
    parser.add_argument('--new-evaluation-dir',type=Path,default=Path('artifacts/evaluations/refiner_aligned_v7_test'))
    parser.add_argument('--data-dir',default='/data_ssd/oyys/eCDI')
    parser.add_argument('--batch-size',type=int,default=12)
    parser.add_argument('--num-workers',type=int,default=4)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Choose a new output directory.')
    if args.batch_size<1 or args.num_workers<0:parser.error('Require batch>=1, workers>=0.')
    reports=[json.loads((d/'provenance.json').read_text()) for d in [args.old_evaluation_dir,args.new_evaluation_dir]]
    for report,path in zip(reports,[args.old_checkpoint,args.new_checkpoint]):
        if report['split']!='test' or report['objective']!=OBJECTIVE_VERSION or report['radius']!=4.:
            raise ValueError('Expected paired V7 test evaluations at search radius 4.')
        if Path(report['checkpoint']).resolve()!=Path(path).resolve():raise ValueError('Checkpoint differs from supplied evaluation.')
        if report['checkpoint_sha256']!=file_digest(path):raise ValueError('Checkpoint contents changed since evaluation.')
    old_scores=read_scores(args.old_evaluation_dir/'evaluation_per_sample.csv')
    new_scores=read_scores(args.new_evaluation_dir/'evaluation_per_sample.csv')
    selected=select_examples(old_scores,new_scores)
    device=torch.device(args.device)
    source=[load_checkpoint(p) for p in [args.old_checkpoint,args.new_checkpoint]]
    if any(s['stage']!='refiner' for s in source):raise ValueError('Expected refiner checkpoints.')
    config=copy.deepcopy(source[0]['config']);config['root']=str(Path(args.data_dir).resolve())
    for key in ['diffraction_kind','input_log_data','center_targets']:
        if source[0]['config'][key]!=source[1]['config'][key]:raise ValueError('Preprocessing mismatch: '+key)
    if source[1].get('registered_protocol',{}).get('search_radius')!=4.:
        raise ValueError('New checkpoint does not record the expected V7 search radius.')
    dataset=build_dataset(config,'test',0,include_ideal=True)
    try:
        for s,report in zip(source,reports):
            verify_data_identity(s,dataset.manifest)
            for key in ['manifest_sha256','config_sha256','selected_indices_sha256']:
                if report['data'][key]!=dataset.manifest[key]:raise ValueError('Evaluation/current data mismatch: '+key)
        if set(dataset.indices)!=set(old_scores):raise ValueError('Score CSV does not cover the complete test split.')
        positions={int(sid):i for i,sid in enumerate(dataset.indices)}
        thickness=read_thickness(dataset)
        models=[make_model('refiner',s['config'],checkpoint=s).to(device).eval() for s in source]
        if state_digest(models[0].phase_model)!=state_digest(models[1].phase_model):raise ValueError('Expected identical frozen phase predictors.')
        loader=DataLoader(Subset(dataset,[positions[i] for i in selected]),batch_size=args.batch_size,shuffle=False,
                          num_workers=args.num_workers,persistent_workers=False,pin_memory=device.type=='cuda')
        args.output_dir.mkdir(parents=True)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
        manifest=[]
        with torch.no_grad():
            for raw in loader:
                batch=prepare_batch(raw,config,device)
                fields=[m(batch['input'],batch['diffraction']).refined_object for m in models]
                aligned=[align_objects(p,batch['realspace'],4.) for p in fields]
                for i in range(len(fields[0])):
                    sid=int(batch['index'][i]);truth=aligned[0][1][i,0].cpu().numpy()
                    predictions={name:values[0][i,0].cpu().numpy() for name,values in zip(['Old refiner','V7 refiner'],aligned)}
                    render(args.output_dir/f'sample_{sid:05d}.png',sid,batch['material'][i],thickness[sid],predictions,truth,selected[sid])
                    manifest.append({'sample':sid,'material':batch['material'][i],'thickness_a':thickness[sid],
                                     'selection':selected[sid],'old_score_csv':float(old_scores[sid]['registered_complex_nrmse']),
                                     'new_score_csv':float(new_scores[sid]['registered_complex_nrmse'])})
                    logging.info('Rendered sample=%d | material=%s',sid,batch['material'][i])
        (args.output_dir/'samples.json').write_text(json.dumps({'old_checkpoint':args.old_checkpoint,'new_checkpoint':args.new_checkpoint,
            'selection_rule':'Baseline median and 90th-percentile error per material, plus two largest regressions. No training or parameter selection.',
            'display':'Whole-image RMS normalized, common amplitude display scale; phase and complex errors after V7 target-based rigid alignment.',
            'samples':manifest},indent=2),encoding='utf-8')
        logging.info('Complete | images=%d | output=%s',len(manifest),args.output_dir)
    finally:
        dataset.close()


if __name__=='__main__':main()
