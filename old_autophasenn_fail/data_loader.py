from typing import Text, TextIO
import json
import numpy as np
import os
import torch
import random


import time

class Dataset(torch.utils.data.Dataset):
    def __init__(self, diff_path, real_path, num_samples, shape_diff=(64, 64, 64), shape_real=(64, 64, 64),
                 dtype_diff='float32', dtype_real='complex64', scale_I=0, shuffle=True):
        self.num_samples = num_samples
        # Kept for CLI compatibility. This memmap loader currently returns raw
        # diffraction data and does not normalize by scale_I.
        self.scale_I = scale_I
        
        self.mmap_diff = np.memmap(diff_path, dtype=dtype_diff, mode='r', 
                                   shape=(num_samples,) + shape_diff)
        self.mmap_real = np.memmap(real_path, dtype=dtype_real, mode='r', 
                                   shape=(num_samples,) + shape_real)
        
        self.indices = list(range(num_samples))
        if shuffle:
            import random
            random.Random(4).shuffle(self.indices)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # --- 计时：磁盘/内存映射读取 ---
        actual_idx = self.indices[index]
        
        # 核心：np.array() 才会真正触发 OS 去磁盘抓取数据
        diff = np.array(self.mmap_diff[actual_idx]) 
        realspace = np.array(self.mmap_real[actual_idx])
        
        # 复数运算通常比较耗时
        amp = np.abs(realspace)
        phi = np.angle(realspace)

        if self.scale_I>0:
            max_I = diff.max()
            diff = diff/(max_I+1e-6)*self.scale_I

        res = (diff[np.newaxis], amp[np.newaxis], phi[np.newaxis])
        
        return res
