import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SRC = Path('/data_ssd/oyys/eCDI/external_test/au5nm_ready_to_use_20261027')
OUT = Path('/data_ssd/oyys/eCDI/external_test/au5nm_ready_512')
OUT.mkdir(exist_ok=True)
psi = np.load(SRC / 'complex_field.npy').astype(np.complex64)
x = torch.from_numpy(np.stack([psi.real, psi.imag])[None])
x = F.interpolate(x, size=(512,512), mode='bilinear', align_corners=False)[0]
psi512 = (x[0].numpy() + 1j*x[1].numpy()).astype(np.complex64)
Fpsi = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psi512.astype(np.complex128)), norm='ortho'))
I512 = (np.abs(Fpsi)**2).astype(np.float32)
np.save(OUT/'realspace.npy', psi512)
np.save(OUT/'diffraction.npy', I512)
config = {'schema_version':1,'dataset_name':'au5nm_ready_paired_512','spatial_dims':2,'shape':[512,512],'root':str(OUT),'diffraction_kind':'intensity','input_log_data':True,'center_targets':True,'splits':{k:{'diffraction':'diffraction.npy','realspace':'realspace.npy'} for k in ('train','val','test')},'training':{'model_variant':'reduced_bn_no_outer_skip','phase':{'epochs':100,'learning_rate':0.001,'scheduler':'plateau'},'refiner':{'epochs':50,'learning_rate':0.0002,'scheduler':'cosine'},'base_channels':4,'num_experts':[3,3,3,3],'top_k':2,'router_heads':2,'router_dropout':0.2,'position_dropout':0.1,'codec_dropout':0.1,'d_state':16,'d_conv':4,'expand':2,'support_threshold':0.3,'physics_loss_weight':0.0,'gradient_clip':1.0}}
(OUT/'config.json').write_text(json.dumps(config,indent=2))
print('wrote', OUT, psi.shape, '->', psi512.shape, 'intensity', float(I512.min()), float(I512.max()))
