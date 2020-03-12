
import numpy as np

from generator import projectPoints, json_load
import os
import math
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from data_generators import get_raw_data,create_image_tensor,create_gaussian

from test_scatter_2 import render_heatmap

from help_functions import read_img

BATCH_SIZE = 16
AUTOTUNE = tf.data.experimental.AUTOTUNE
gaussian = np.zeros((12,12))
radius = 1

def init_gaussian():
    std = 2
    global gaussian
    gaussian = create_gaussian(std, radius)
    print('GAUSSIAN')
    print(gaussian)

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

def gen_function(xyz_list, K_list):
    data = []
    num_samples = 10
    image_tensor = create_image_tensor(dir_path, indicies, num_samples,batch_size=16)
    i = 0
    output_shape = (224,224)
    num_keypoints = 21
    batch_size = BATCH_SIZE
    init_gaussian()
    for i in range(10):
        batch_x = []
        batch_y = [[], [], []]
        coords = np.zeros((batch_size, num_keypoints, 2))
        for j in range(batch_size):
            idx = indicies[i + j]
            img = read_img(idx, dir_path, 'training') / 255.0
            uv = projectPoints(xyz_list[i + j], K_list[i + j])
            coords[j, :, :] = uv
            # hm = create_gaussian_hm(uv, 224, 224, radius, hm_small)
            batch_x.append(img)
            # batch_y[0].append(hm[0])
            # batch_y[1].append(hm[1])
            # batch_y[2].append(hm[2])

            if i + j == num_samples - 1:
                i = -j
        i += batch_size
        hm1 = render_heatmap(coords, [x // 4 for x in output_shape], num_keypoints, gaussian)
        hm2 = render_heatmap(coords, [x // 2 for x in output_shape], num_keypoints, gaussian)
        hm3 = render_heatmap(coords, output_shape, num_keypoints, gaussian)

        #  imgs = sess.run(hm1)
        batch_y[0] = tf.data.Dataset.from_tensor_slices(hm1).batch(
            batch_size)  # tf.reshape(hm1, [batch_size, num_keypoints, -1])
        batch_y[1] = tf.data.Dataset.from_tensor_slices(hm2).batch(
            batch_size)  # tf.reshape(hm1, [batch_size, num_keypoints, -1])
        batch_y[2] = tf.data.Dataset.from_tensor_slices(hm3).batch(batch_size)

        yield (batch_x,batch_y)




if __name__ == '__main__':
    dir_path = sys.argv[1]
    xyz_list, K_list, indicies, num_samples = get_raw_data(dir_path, 'training')

    #image_tensor = tf.data.Dataset.zip((image_tensor))
   # print(image_tensor)
   # ds = prepare_for_training(image_tensor)
    #for e in image_tensor:
      #  print(e)
    dataset = tf.data.Dataset.from_generator(gen_function, (tf.float32, tf.float32), args=([xyz_list, K_list]))
    print(dataset)
    for d in dataset:
        print(d)