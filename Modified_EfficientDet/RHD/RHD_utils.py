import os
import sys
import pickle
import json

sys.path.insert(1, '../')


from help_functions import *


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg



def read_img_RHD(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'color',
                                '%05d.png' % sample_version.map_id(idx, version))
    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)

def get_RHD_coords_2d(dir_path, data_set):
    coords = []
    if data_set == 'training':
        data_path = os.path.join(dir_path,'training', 'anno_training.pickle')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        for idx, sample in data.items():
            print(len(sample['uv_vis'][:,:2]))
            break
            coords.append(sample['uv_vis'][:,:2])

    return np.array(coords)


def dataGenerator_RHD(dir_path, batch_size = 16, data_set = 'training'):
    idx = 1
    if data_set == 'training':
        labels = get_RHD_coords_2d(dir_path, 'training')
        max_idx = 41257-1000
    elif data_set == 'validation':
        labels = get_RHD_coords_2d(dir_path, 'training')
        max_idx = 41257
        idx = 41257-1000
    elif data_set == 'evaluation':
        max_idx = 2727

    while True:
        batch_x = []
        batch_y = [[], [], []]
        j = 0

        while j < batch_size:
            
            try:      
                coords = labels[idx]
                img = read_img_RHD(idx, dir_path, data_set)/255
                onehots = create_onehot(coords, 80,80)
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
                if data_set == 'validation':
                    idx = 41257-1000
                else:
                    idx = 1


        yield (np.array(batch_x), batch_y)


