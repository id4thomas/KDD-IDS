import os
import h5py

import pandas as pd
import numpy as np

def get_hdf5_data(file_path,labeled=False):
    with h5py.File(file_path,'r') as f:
        data=f['x'].value
        if labeled:
            label=f['y'].value
        else:
            label=[]
    return data,label
