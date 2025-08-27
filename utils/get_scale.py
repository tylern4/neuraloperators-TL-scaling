import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import h5py



datapath = "."
with h5py.File(os.path.join(datapath, "train_k1_5_32k.h5"), "r") as f:
    x_train = f['fields'][:]
    x_tensor = f['tensor'][:]
    print(list(f.keys()))
source_norm = []
sol_max = []
tensor_max = []
nx = ny = 128
lx = ly = 1
num_ten = x_tensor.shape[1]

for i in range(x_train.shape[0]):
    sn = np.linalg.norm(x_train[i,0]) * lx/nx * ly/ny
    source_norm.append(sn)
    sol_max.append(np.max(np.abs(x_train[i,1])))
    tensor_max.append([np.abs(x_tensor[i,t_idx]) for t_idx in range(num_ten)])

tensor_max = np.array(tensor_max)
source_scale = np.median(source_norm)
sol_scale = np.median(sol_max)
tensor_scale = [np.median(tensor_max[:,j]) for j in range(num_ten)]

scale = [source_scale] + tensor_scale + [sol_scale] + [lx, ly]
print(scale)
np.save(os.path.join(datapath, "train_k1_5_scales.npy"), scale)
