import numpy as np
from pathlib import Path

root = Path('/home/oyys/code/UMamba-AutoPhaseNN/high_strain_CNN/artifacts/evaluations/au5nm_diag_b8')
for p in sorted(root.glob('*.npz')):
    z = np.load(p)
    print(p.name, z.files)
    for k in z.files:
        a = z[k]
        if np.iscomplexobj(a):
            print(k, a.shape, a.dtype, 'abs', float(np.abs(a).min()), float(np.abs(a).max()), 'meanabs', float(np.abs(a).mean()))
        elif np.issubdtype(a.dtype, np.number):
            print(k, a.shape, a.dtype, float(a.min()), float(a.max()), float(a.mean()))
        else:
            print(k, a.shape, a.dtype, a.tolist())
