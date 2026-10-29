from pathlib import Path
import numpy as np

p = Path('/data_ssd/oyys/eCDI/external_test/au5nm_ready_512')
for name in ('diffraction.npy', 'realspace.npy'):
    a = np.load(p / name)
    print(name, a.shape, a.dtype)
    if a.ndim == 2:
        np.save(p / name, a[None])
    elif a.ndim != 3:
        raise ValueError(a.shape)
for name in ('diffraction.npy', 'realspace.npy'):
    a = np.load(p / name)
    print('fixed', name, a.shape, a.dtype)
