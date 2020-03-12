import tensorflow as tf

from test_scatter_2 import render_heatmap

from generator import json_load,projectPoints
import numpy as np

from data_generators import create_gaussian
from help_functions_tmp import read_img
import os

if __name__ == '__main__':
    dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"  #
    imgs = []
    batch_size = 2
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    uv = []
    batch_size = 2
    num_keypoints = 21

    coords = np.zeros((batch_size, num_keypoints, 2))

    for i in range(batch_size):# range(len(xyz_list)):
        uv_i = projectPoints(xyz_list[i], K_list[i])
        uv_i = uv_i[:, ::-1]
        uv.append(uv_i)
        coords[i, :, :] = uv_i
    for j in range(batch_size):
        img = read_img(j, dir_path, 'training') / 255.0
        imgs.append(img)
    output_shape = (224,224)
    gaussian = create_gaussian(2, 1)

    hm = render_heatmap(coords, output_shape, num_keypoints, gaussian)

    # Two tensors can be combined into one Dataset object.
    labels = hm#tf.constant(hm, shape=(batch_size, 224, 224,21))
    features = tf.constant(imgs)
    dataset = tf.data.Dataset.from_tensors(labels)


    #for element in dataset.as_numpy_iterator():
    #    print(element)


