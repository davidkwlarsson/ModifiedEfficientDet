import os
import sys
import json
import random

import numpy as np
import tensorflow as tf

sys.path.insert(1, '../')

from help_functions import *




def dataGenerator(dir_path, batch_size = 16, data_set = 'training', shuffle = True):
    if data_set == 'training':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))#[:-560]
        xyz_list *= 4
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))#[:-560]
        K_list *= 4
        indicies = [i for i in range(32000)] + [i for i in range(32560,64560)] + [i for i in range(65120,97120)] + [i for i in range(97680,129680)]
        num_samples = len(indicies)
        print("Total number of training samples: ", num_samples, " and ", len(indicies))
    elif data_set == 'validation':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))#[-560:]
        xyz_list *= 4
        # num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))#[-560:]
        K_list *= 4
        indicies = [i for i in range(32000,32560)] + [i for i in range(64560,65120)] + [i for i in range(97120, 97680)] + [i for i in range(129680,130240)]
        num_samples = len(indicies)
        print("Total number of validation samples: ", num_samples," and ", len(indicies))
    elif data_set == 'evaluation':
        xyz_list = json_load(os.path.join(dir_path, 'evaluation_xyz.json')  )
        num_samples = len(xyz_list)
        print("Total number of evaluation samples: ", num_samples)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))

    else:
        print("No specified data found!")
        sys.exit()
        

    i = 0
    while True:
        batch_x = []
        batch_y = [[],[]]
        j = 0
        nbr_samp = 0
        while j < batch_size:
            idx = indicies[i+j]
            img = read_img(idx, dir_path, data_set)/255
            uv = projectPoints(xyz_list[idx], K_list[idx])
            # depthmaps = get_depthmaps(uv, xyz_list[idx])
            try:
                onehots = create_onehot(uv, 28,28)
                depth = get_depth(xyz_list[idx])
                batch_x.append(img)
                batch_y[0].append(onehots)
                batch_y[1].append(depth)
                nbr_samp +=1
                
            except:
                print('invalid image, coordinates out of range')
            j += 1
            # batch_x.append(img)
            # batch_y[0].append(onehots)
            # batch_y[1].append(onehots[1])
            # batch_y[2].append(onehots[2])
            # batch_y[3].append(depthmaps)
            # batch_y[1].append(depth)
            if i+j == num_samples-1:
                i = -j
                if shuffle:
                    random.shuffle(indicies)
        i += batch_size

        yield (np.array(batch_x), batch_y)