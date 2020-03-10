
# import tensorflow as tf

# tf.test.is_gpu_available  
# tf.test.gpu_device_name

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# assert tf.test.is_gpu_available()
# assert tf.test.is_built_with_cuda()

# print(tf.test.is_built_with_cuda())

# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
#     print(c)

import pickle
import os
import sys

import numpy as np
from FreiHAND.freihand_utils import dataGenerator
from help_functions import *

def try_obman(dir_path):
    data_path = os.path.join(dir_path,'train', 'meta', '%08d.pkl' % 1)
    print(data_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(data.keys())
    coords = data['coords_2d']
    print(coords)

def try_RHD(dir_path):
    # This loads a dict containing dicts. The first dict has the index as keys
    # and the second provide coordinates in 3d, 2d and the camera matrix

    data_path = os.path.join(dir_path,'training', 'anno_training.pickle')
    print(data_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(data[1].keys())

    print(data[1]['uv_vis'][:,:2])
    i = 0
    for idx, sample in data.items():
        print(sample['uv_vis'][:,:2])
        i += 1
        if i == 10:
            break
        
def try_plotting_stuff(dir_path):

    traingen = dataGenerator(dir_path, batch_size = 16, data_set = 'training')
    validgen = dataGenerator(dir_path, batch_size= 16, data_set = 'validation')

    validgen2 = dataGenerator(dir_path, batch_size= 10, data_set = 'validation')
    (images, targets) = next(validgen2)

    
    (heatmaps, heatmaps2, heatmaps3, depth) = targets
    (preds, preds2 ,preds3, depth_pred) = targets
    # (heatmaps, heatmaps2, heatmaps3) = targets
    # (preds,preds2, preds3) = targets

    coord_preds = heatmaps_to_coord(preds3)
    coord = heatmaps_to_coord(heatmaps3)
    
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
    K_list = K_list[:len(preds3)]

    xyz_list = add_depth_to_coords(coord, depth, K_list)
    # xy = []
    # for i in range(len(coord)):
    #     xy.append(np.array([coord[i][0::2], coord[i][1::2]]).T)
    
    save_coords(xyz_list, images[0])

    # plot_predicted_heatmaps(preds, heatmaps, images)
    # plot_predicted_hands_uv(images, coord_preds*4)
    # plot_predicted_coordinates(images, coord_preds*4, coord*4)

import tensorflow as tf

import time
default_timeit_steps = 1000
BATCH_SIZE = 16

def timeit(ds, steps=default_timeit_steps):
  start = time.time()
  it = iter(ds)
  for i in range(steps):
    batch = next(it)
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))

def try_tfdata(dir_path):
    print(tf.__version__)
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    length = len(xyz_list)
    # xyz_list *= 4
    xyz_data = tf.data.Dataset.from_tensor_slices(xyz_list)
    
    image_path = os.path.join(dir_path, 'training/rgb/*')
    list_ds = tf.data.Dataset.list_files(image_path, shuffle = False)
    # for f in list_ds.take(5):
    #     print(f.numpy())

    list_ds = list_ds.take(length)

    def get_tfimage(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img,channels = 3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224,224])
        return img
    list_ds = list_ds.map(get_tfimage)

    # for image in list_ds.take(1):
    #     print("Image shape: ", image.numpy().shape)

    labeled_ds = tf.data.Dataset.zip((list_ds, xyz_data))
    labeled_ds = labeled_ds.shuffle(buffer_size = length)
    labeled_ds = labeled_ds.repeat()
    labeled_ds = labeled_ds.batch(16)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    labeled_ds = labeled_ds.prefetch(buffer_size = AUTOTUNE)
    # labeled_ds = labeled_ds.cache(filename) # Save to memory first time running trough then access that.
    # for image,label in labeled_ds.take(1):
    #     print("Image shape: ", image.numpy().shape)
    #     print(label)
    timeit(labeled_ds)

def main():
    dir_path = sys.argv[1]
    # try_obman(dir_path)
    # try_RHD(dir_path)
    # try_plotting_stuff(dir_path)
    try_tfdata(dir_path)


if __name__ == '__main__':
    # tf.keras.backend.clear_session()
    main()
