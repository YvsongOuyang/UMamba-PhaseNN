"""Refiner pilot using one registered-complex objective for training/scoring.

Starts from the incumbent complete cascade. Phase parameters AND BatchNorm
buffers remain frozen. Epoch zero is reevaluated under V7's own convention and
eligible for best. Source files/checkpoints are never overwritten. No new data,
architecture, phase checkpoint substitution, or auxiliary losses are introduced.
"""
from __future__ import annotations
import argparse
import copy
import csv
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
from .data_v2 import build_dataset,prepare_batch
from .train import load_checkpoint,make_model,verify_data_identity
from .aligned_objective_v7 import OBJECTIVE_VERSION,objective_and_metrics

LOGGER=logging.getLogger(__name__)
VERSION='registered_refiner_v7'
PROJECT_DIR=Path(__file__).resolve().parents[1]


def state_digest(module):
    digest=hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        tensor=tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read_thickness(dataset):
    import h5py
    with h5py.File(dataset.path,'r') as f:
        metadata=f['metadata_json'].asstr()
        return {int(i):float(json.loads(metadata[i])['structure']['target_thickness_a']) for i in dataset.indices}


def summarize(records):
    keys=[k for k in records[0] if k not in ('sample','stage','material','thickness_a')]
    def mean(rows):
        return {'num_samples':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in keys}} if rows else {'num_samples':0}
    result={}
    for stage in sorted({r['stage'] for r in records}):
        selected=[r for r in records if r['stage']==stage]
        result[stage]={'all':mean(selected),'thin_le20':mean([r for r in selected if r['thickness_a']<=20]),
                       'thick_ge34':mean([r for r in selected if r['thickness_a']>=34]),
                       'by_material':{m:mean([r for r in selected if r['material']==m]) for m in sorted({r['material'] for r in selected})}}
    return result


def save_results(root,name,summary,records):
    (root/f'{name}_metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    with (root/f'{name}_per_sample.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


@torch.no_grad()
def evaluate(model,loader,config,device,thickness,radius):
    model.eval()
    records=[]
    for index,raw in enumerate(loader,1):
        batch=prepare_batch(raw,config,device)
        output=model(batch['input'],batch['diffraction'])
        for stage,field in [('initial',output.initial_object),('refined',output.refined_object)]:
            _,values=objective_and_metrics(field,batch['realspace'],radius)
            arrays={k:v.detach().cpu().tolist() for k,v in values.items()}
            for i in range(len(field)):
                sid=int(batch['index'][i])
                records.append({'sample':sid,'stage':stage,'material':batch['material'][i],'thickness_a':thickness[sid],**{k:v[i] for k,v in arrays.items()}})
        if index%20==0 or index==len(loader):LOGGER.info('VAL | batch=%d/%d',index,len(loader))
    if len(records)!=2*len(loader.dataset):raise RuntimeError('Validation incomplete.')
    for stage in ('initial','refined'):
        if len({r['sample'] for r in records if r['stage']==stage})!=len(loader.dataset):
            raise RuntimeError('Validation sample identities are not unique.')
    if len({r['material'] for r in records})!=6:raise RuntimeError('Expected all six materials in validation.')
    return summarize(records),records


def train_epoch(model,loader,config,device,optimizer,radius):
    model.train()
    if model.phase_model.training:raise RuntimeError('Frozen phase must remain in eval mode.')
    totals={};count=0;start=time.monotonic()
    for index,raw in enumerate(loader,1):
        batch=prepare_batch(raw,config,device)
        optimizer.zero_grad(set_to_none=True)
        output=model(batch['input'],batch['diffraction'])
        loss,values=objective_and_metrics(output.refined_object,batch['realspace'],radius)
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite registered loss.')
        loss.backward()
        parameters=[p for p in model.refiner.parameters() if p.requires_grad and p.grad is not None]
        if not parameters:raise RuntimeError('Refiner has no gradients.')
        clip=config['training'].get('gradient_clip',1.)
        if clip>0:torch.nn.utils.clip_grad_norm_(parameters,clip,error_if_nonfinite=True)
        elif any(not bool(torch.isfinite(p.grad).all()) for p in parameters):raise ValueError('Nonfinite gradient.')
        optimizer.step()
        n=len(output.refined_object);count+=n
        for k,v in {'loss':loss.detach(),**{k:v.mean() for k,v in values.items()}}.items():
            totals[k]=totals.get(k,0.)+float(v)*n
        if index%50==0 or index==len(loader):
            elapsed=time.monotonic()-start
            LOGGER.info('TRAIN | batch=%d/%d | registered_loss=%.6f | elapsed=%.1fs | eta=%.1fs',index,len(loader),float(loss),elapsed,elapsed/index*(len(loader)-index))
    if count!=len(loader.dataset):raise RuntimeError('Training split incomplete.')
    return {k:v/count for k,v in totals.items()}


def rng_state(generators):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'loaders':{k:g.get_state() for k,g in generators.items()}}


def restore_rng(state,generators):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])
    for k,g in generators.items():g.set_state(state['loaders'][k])


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-refiner-checkpoint');source.add_argument('--resume');source.add_argument('--evaluate-checkpoint')
    parser.add_argument('--data-dir',default='/data_ssd/oyys/eCDI')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--stop-after-epochs',type=int,default=1)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--search-radius',type=float,default=4.)
    parser.add_argument('--batch-size',type=int,default=12)
    parser.add_argument('--num-workers',type=int,default=4)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--run-name',default='')
    parser.add_argument('--checkpoint-root',type=Path,default=Path('/data_ssd/oyys/high_strain_cnn/pipeline_output_2d'))
    parser.add_argument('--runs-dir',type=Path,default=PROJECT_DIR/'artifacts/training/ecdi_training')
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if not 1<=args.stop_after_epochs<=args.epochs<=3:parser.error('Require 1 <= stop-after-epochs <= epochs <= 3.')
    if args.batch_size<1 or args.num_workers<0:parser.error('Require batch>=1, workers>=0.')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=1e-6:parser.error('Require finite learning-rate > 1e-6.')
    if not math.isfinite(args.search_radius) or not 0<args.search_radius<=8:parser.error('Require finite 0 < search-radius <= 8.')
    if args.evaluate_checkpoint and args.output_dir is None:parser.error('Evaluation requires --output-dir.')
    if not args.evaluate_checkpoint and args.split!='val':parser.error('Training selects models on val only.')
    if not args.evaluate_checkpoint and args.output_dir:parser.error('Use --runs-dir for training output.')
    return args


def main():
    args=parse_args()
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable.')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    path=args.resume or args.init_refiner_checkpoint or args.evaluate_checkpoint
    source=load_checkpoint(path)
    if source.get('stage')!='refiner':raise ValueError('A native refiner checkpoint is required.')
    config=copy.deepcopy(source['config'])
    if config.get('storage')!='ecdi_hdf5' or config['training'].get('phase_architecture','baseline')!='baseline':
        raise ValueError('V7 expects the native eCDI baseline cascade.')
    config['root']=str(Path(args.data_dir).expanduser().resolve())
    protocol={'epochs':args.epochs,'learning_rate':args.learning_rate,'search_radius':args.search_radius,
              'batch_size':args.batch_size,'num_workers':args.num_workers,'seed':args.seed,'device':args.device,
              'objective':OBJECTIVE_VERSION}
    if args.resume and (source.get('experiment_version')!=VERSION or source.get('registered_protocol')!=protocol):
        raise ValueError('Resume requires the same V7 training protocol.')
    if args.evaluate_checkpoint and source.get('experiment_version')==VERSION:
        if source['registered_protocol']['search_radius']!=args.search_radius:raise ValueError('Evaluation must use the saved training search radius.')
    name=args.run_name or f'refiner_aligned_v7_{datetime.now():%Y%m%d_%H%M%S}'
    if Path(name).name!=name or name in ('.','..'):raise ValueError('Run name must be one directory name.')
    root=args.output_dir if args.evaluate_checkpoint else args.runs_dir/name
    ckpt_dir=args.checkpoint_root/name
    if args.resume:
        if Path(path).resolve().parent!=ckpt_dir.resolve() or not root.is_dir():raise ValueError('Resume must use the original run name and paths.')
        if source['epoch']>=args.stop_after_epochs:raise ValueError('Stop epoch must exceed resumed epoch.')
    elif root.exists() or (not args.evaluate_checkpoint and ckpt_dir.exists()):
        raise FileExistsError('Choose a fresh run/output directory; previous results are preserved.')
    datasets={};loaders={};generators={}
    try:
        for split in ([args.split] if args.evaluate_checkpoint else ['train','val']):
            datasets[split]=build_dataset(config,split,0,include_ideal=split!='train')
            verify_data_identity(source,datasets[split].manifest)
            for k in ['config_sha256','manifest_sha256']:
                if not source.get('data_manifests',{}).get('train',{}).get(k):raise ValueError('Source checkpoint lacks data identity hashes.')
            if datasets[split].manifest['selected_samples']!=datasets[split].manifest['available_samples']:
                raise ValueError('Use the complete immutable split.')
            previous=source.get('data_manifests',{}).get(split,{})
            if previous.get('selected_indices_sha256') and previous['selected_indices_sha256']!=datasets[split].manifest['selected_indices_sha256']:
                raise ValueError('Source/current sample identities differ.')
            generators[split]=torch.Generator().manual_seed(args.seed+(split!='train'))
            loaders[split]=DataLoader(datasets[split],batch_size=args.batch_size,shuffle=split=='train',num_workers=args.num_workers,
                generator=generators[split],persistent_workers=False,pin_memory=device.type=='cuda',drop_last=False)
        val_split=args.split if args.evaluate_checkpoint else 'val'
        thickness=read_thickness(datasets[val_split])
        model=make_model('refiner',config,checkpoint=source).to(device)
        frozen_hash=state_digest(model.phase_model)
        if any(p.requires_grad for p in model.phase_model.parameters()):raise RuntimeError('Phase is not frozen.')
        root.mkdir(parents=True,exist_ok=bool(args.resume))
        logging.basicConfig(level=logging.INFO,force=True,format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[logging.StreamHandler(),logging.FileHandler(root/'console.log',encoding='utf-8')])
        LOGGER.info('Experiment=%s | objective=%s | radius=%.3f | batch=%d',VERSION,OBJECTIVE_VERSION,args.search_radius,args.batch_size)
        if args.evaluate_checkpoint:
            summary,records=evaluate(model,loaders[val_split],config,device,thickness,args.search_radius)
            save_results(root,'evaluation',summary,records)
            (root/'provenance.json').write_text(json.dumps({'checkpoint':str(Path(path).resolve()),'checkpoint_sha256':file_digest(path),
                'objective':OBJECTIVE_VERSION,'radius':args.search_radius,'split':val_split,'data':datasets[val_split].manifest},indent=2),encoding='utf-8')
            LOGGER.info('Evaluation complete | registered_complex_nrmse=%.6f',summary['refined']['all']['registered_complex_nrmse'])
            return 0
        config['experiment_version']=VERSION
        config['training']['refiner']={'objective':OBJECTIVE_VERSION,'learning_rate':args.learning_rate,'epochs':args.epochs,'scheduler':'cosine'}
        optimizer=torch.optim.Adam(model.refiner.parameters(),lr=args.learning_rate)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
        ckpt_dir.mkdir(parents=True,exist_ok=bool(args.resume))
        if args.resume:
            if source['frozen_phase_sha256']!=frozen_hash:raise ValueError('Frozen phase state differs.')
            optimizer.load_state_dict(source['optimizer_state_dict']);scheduler.load_state_dict(source['scheduler_state_dict'])
            history=copy.deepcopy(source['history']);best=source['best_val_metric'];reference=source['reference_metrics'];lineage=source['lineage']
            restore_rng(source['random_states'],generators);start=source['epoch']+1
        else:
            history=[];best=float('inf');reference=None;start=1
            lineage={'source_checkpoint':str(Path(path).resolve()),'source_sha256':file_digest(path),'source_epoch':source['epoch']}
        manifests={s:d.manifest for s,d in datasets.items()}
        (root/'config.json').write_text(json.dumps({'config':config,'protocol':protocol,'lineage':lineage,'data_manifests':manifests},indent=2),encoding='utf-8')

        def validate_save(epoch,train=None):
            nonlocal best,reference
            if state_digest(model.phase_model)!=frozen_hash:raise RuntimeError('Frozen phase parameters or BatchNorm changed.')
            summary,records=evaluate(model,loaders['val'],config,device,thickness,args.search_radius)
            if state_digest(model.phase_model)!=frozen_hash:raise RuntimeError('Evaluation changed frozen phase state.')
            values=summary['refined']['all'];metric=values['registered_complex_nrmse']
            if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite validation metrics.')
            if reference is None:
                reference=copy.deepcopy(summary);save_results(root,'reference',summary,records)
            improved=metric<best;best=min(best,metric)
            history.append({'epoch':epoch,'train':train,'val':summary,'lr':optimizer.param_groups[0]['lr']})
            saved={'experiment_version':VERSION,'spatial_dims':2,'stage':'refiner','epoch':epoch,'config':config,
                'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                'best_val_metric':best,'checkpoint_selection_metric':'registered_complex_nrmse','registered_protocol':protocol,
                'reference_metrics':reference,'validation_metrics':summary,'lineage':lineage,'history':history,'data_manifests':manifests,
                'frozen_phase_sha256':frozen_hash,'random_states':rng_state(generators)}
            torch.save(saved,ckpt_dir/'checkpoint_last.pt')
            if improved:
                torch.save(saved,ckpt_dir/'checkpoint_best.pt');save_results(root,'best',summary,records)
            save_results(root,'last',summary,records)
            (root/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False),encoding='utf-8')
            LOGGER.info('Epoch=%d/%d | registered_nrmse=%.6f | best=%.6f | reference=%.6f | amplitude_rms_error=%.6f | thick_nrmse=%.6f | offaxis_nrmse=%.6f | boundary_fraction=%.3f | lr=%.3g',
                epoch,args.epochs,metric,best,reference['refined']['all']['registered_complex_nrmse'],values['amplitude_relative_l2_rms'],
                summary['refined']['thick_ge34'].get('registered_complex_nrmse',float('nan')),values['offaxis_nrmse_9x9'],values['search_boundary_hit'],optimizer.param_groups[0]['lr'])
        if not args.resume:validate_save(0)
        LOGGER.info('Only refiner is optimized | one registered complex loss | horizon=%d | stop=%d',args.epochs,args.stop_after_epochs)
        for epoch in range(start,args.stop_after_epochs+1):
            train=train_epoch(model,loaders['train'],config,device,optimizer,args.search_radius)
            scheduler.step();validate_save(epoch,train)
        LOGGER.info('Pilot stopped | best=%.6f | reference=%.6f | source checkpoint preserved=%s',best,reference['refined']['all']['registered_complex_nrmse'],lineage['source_checkpoint'])
    finally:
        for dataset in datasets.values():dataset.close()
    return 0


if __name__=='__main__':main()
