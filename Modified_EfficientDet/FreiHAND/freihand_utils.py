import os
import sys
import json

import numpy as np

sys.path.insert(1, '../')

from help_functions import *




def dataGenerator(dir_path, batch_size = 16, data_set = 'training'):
    if data_set == 'training':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[:-560]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[:-560]
        K_list *= 4
        indicies = [i for i in range(32000)] + [i for i in range(32560,64560)] + [i for i in range(65120,97120)] + [i for i in range(97680,129680)]
        print("Total number of training samples: ", num_samples, " and ", len(indicies))
    elif data_set == 'validation':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
        K_list *= 4
        indicies = [i for i in range(32000,32560)] + [i for i in range(64560,65120)] + [i for i in range(97120, 97680)] + [i for i in range(129680,130240)]
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
        batch_y = [[], [], []]
        for j in range(batch_size):
            idx = indicies[i+j]
            img = read_img(idx, dir_path, 'training')/255
            uv = projectPoints(xyz_list[i+j], K_list[i+j])
            # depthmaps = get_depthmaps(uv, xyz_list[idx])
            depth = get_depth(xyz_list[i+j])
            onehots = create_onehot(uv, 56,56)
            batch_x.append(img)
            batch_y[0].append(onehots[0])
            batch_y[1].append(onehots[1])
            batch_y[2].append(onehots[2])
            # batch_y[3].append(depthmaps)
            # batch_y[3].append(depth)
            if i+j == num_samples-1:
                i = -j
        i += batch_size

        yield (np.array(batch_x), batch_y)