import os
import sys
import pickle
import json
import numpy as np

sys.path.insert(1, '../')
from help_functions import *


def get_obman_coords_2d(idx, dir_path, data_set):
    data_path = os.path.join(dir_path, data_set , 'meta', '%08d.pkl' % idx)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data['coords_2d']


def dataGenerator_obman(dir_path, batch_size = 16, data_set = 'train'):
    idx = 1
    if data_set == 'train':
        max_idx = 199998
    elif data_set == 'val':
        max_idx = 9999

    while True:
        batch_x = []
        batch_y = [[], [], []]
        j = 0

        while j < batch_size:
            try:
                
                coords = get_obman_coords_2d(idx, dir_path, data_set)
                img = read_img(idx, dir_path, data_set)/255
                onehots = create_onehot(coords, 64,64)
                batch_x.append(img)
                batch_y[0].append(onehots[0])
                batch_y[1].append(onehots[1])
                batch_y[2].append(onehots[2])
                j += 1
                idx += 1
            except:
                idx += 1
                
                continue
        
            
            if idx == max_idx:
                idx = 1

        yield (np.array(batch_x), batch_y)


