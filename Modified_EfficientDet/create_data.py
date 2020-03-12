import argparse
#from utils.fh_utils import *
import numpy as np
import os
import math
from help_functions import create_gaussian_hm, create_gaussian_blob

from generator import json_load, projectPoints
from numpy.lib.format import open_memmap

def create_heatmaps(dir_path, num_samples):

    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    heatmaps = []
    print('len of list: ', len(xyz_list))
    for i in range(len(xyz_list)):
        uv = projectPoints(xyz_list[i], K_list[i])
        uv = uv[:, ::-1]

        hm_small = create_gaussian_blob(2, 6)
        hm = create_gaussian_hm(uv, 224, 224, 6, hm_small)
        heatmaps.append(hm)

    return np.array(heatmaps)


def write_heatmaps(heatmaps, num_samples, image_size, dir_path):
'''THIS USED WAY TO MUCH MEMORY'''
    channels = 21
    target_shape = (num_samples, image_size[0],image_size[1], channels)
    targets = open_memmap(dir_path+'/training/hm/heatmaps.npy', dtype=np.float32, mode='w+', shape=target_shape)

    for n in range(num_samples):
        targets[n] = np.transpose(heatmaps[n], (1,2,0))


def read_heatmaps(dir_path):

    target = np.load(dir_path+'/training/hm/heatmaps.npy', mmap_mode='r')

    return target



#def tiny_network():