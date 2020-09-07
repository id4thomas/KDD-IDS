import os
import h5py

import pandas as pd
import numpy as np

def get_hdf5_data(file_path,labeled=False,only_safe=False):
    with h5py.File(file_path,'r') as f:
        data=f['x'].value
        if labeled:
            label=f['y'].value
        else:
            label=[]
    if only_safe:
        safe_idx=[label==0]
        print("safe {} total {}".format(data[safe_idx].shape[0],data.shape[0]))
        data=data[tuple(safe_idx)]
        label=label[tuple(safe_idx)]
    return data,label
