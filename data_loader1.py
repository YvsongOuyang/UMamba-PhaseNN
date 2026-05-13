from typing import Text, TextIO
import json
import numpy as np
import os
import torch
import random


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_ID, data_path, dataset='all', load_all=False, scale_I=0, shuffle=True):
        'Initialization'

            
        pos = list(range(len(data_ID)))
        if shuffle:
            random.Random(4).shuffle(pos)
        self.data_ID = [data_ID[k] for k in pos]
                
        self.data_path = data_path
        self.load_all = load_all
        self.scale_I = scale_I
        if self.load_all:
            """
                if load all data, initialize the database
                load all data will take a lot of memory.
            """
            data_folder = self.data_path
            diff_list = []
            amp_list = []
            phi_list = []
            for img_n in self.data_ID:

                diff = np.load(self.data_path+img_n)['arr_0']
                realspace = np.load(self.data_path+img_n)['arr_1']
                amp = np.abs(realspace)
                phi = np.angle(realspace)
                
                if self.scale_I>0:
                    max_I = diff.max()
                    diff = diff/max_I*self.scale_I

                diff_list.append(diff[np.newaxis])
                amp_list.append(amp[np.newaxis])
                phi_list.append(phi[np.newaxis])
            
            self.diff_list = diff_list
            self.amp_list = amp_list
            self.phi_list = phi_list

            print('All data loaded')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_ID)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.load_all:
            return np.array(self.diff_list[index]),\
                   np.array(self.amp_list[index]), \
                   np.array(self.phi_list[index])
        else:
            # Select sample
            img_ID = self.data_ID[index]
            data_folder = self.data_path
            data = np.load(os.path.join(self.data_path, img_ID))
            diff = data['arr_0']
            realspace = data['arr_1']
            amp = np.abs(realspace)
            phi = np.angle(realspace)


            return diff[np.newaxis], amp[np.newaxis], phi[np.newaxis]


