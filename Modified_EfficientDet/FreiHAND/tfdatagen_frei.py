import sys
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

sys.path.insert(1, '../')

from heatmapsgen import projectPoints
from help_functions import json_load


def create_image_dataset(dir_path, num_samples, data_set):
    image_path = os.path.join(dir_path, data_set ,'rgb/*')
    list_ds = tf.data.Dataset.list_files(image_path, shuffle=False)

    list_ds = list_ds.take(num_samples)

    def get_tfimage(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224])
        return img

    list_ds = list_ds.map(get_tfimage)
    return list_ds

def get_raw_data(dir_path, data_set):
    data_set = data_set.decode('utf8')
    dir_path = dir_path.decode('ascii')
    if data_set == 'training':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[:-560]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[:-560]
        K_list *= 4
        indicies = [i for i in range(32000)] + \
                   [i for i in range(32560, 64560)] + \
                   [i for i in range(65120, 97120)] + \
                   [i for i in range( 97680, 129680)]
        print("Total number of training samples: ", num_samples, " and ", len(indicies))
    elif data_set == 'validation':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
        K_list *= 4
        indicies = [i for i in range(32000, 32560)] + \
                   [i for i in range(64560, 65120)] + \
                   [i for i in range(97120, 97680)] + \
                   [i for i in range(129680,130240)]
        print("Total number of validation samples: ", num_samples, " and ", len(indicies))
    elif data_set == 'evaluation':
        xyz_list = json_load(os.path.join(dir_path, 'evaluation_xyz.json'))
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))
        indicies = 0

    else:
        print("No specified data found!")
        sys.exit()


    return xyz_list, K_list, indicies, num_samples

def tf_render_heatmap(coord, output_shape,num_keypoints, gaussian):
    batch_size = 1  # tf.shape(coord)[0]
    d = 9
    print(np.shape(coord))
    gaussian = tf.repeat(gaussian, batch_size * num_keypoints, axis=1)
    input_shape = output_shape
    x = tf.reshape(coord[:, 0] / input_shape[1] * output_shape[1], [-1])
    y = tf.reshape(coord[:, 1] / input_shape[0] * output_shape[0], [-1])
    x_floor = tf.floor(x) - 1
    y_floor = tf.floor(y) - 1
    x_top = tf.floor(x) + 1
    y_top = tf.floor(y) + 1

    x_floor = tf.clip_by_value(x_floor, 3, output_shape[1] - 3)  # fix out-of-bounds x
    y_floor = tf.clip_by_value(y_floor, 3, output_shape[0] - 3)  # fix out-of-bounds y
    indices_batch = tf.expand_dims(tf.cast( \
        tf.reshape(
            tf.transpose( \
                tf.tile( \
                    tf.expand_dims(tf.range(batch_size), 0) \
                    , [num_keypoints, 1]) \
                , [1, 0]) \
            , [-1]), dtype=tf.float32), 1)
    indices_batch = tf.concat([indices_batch] * d, axis=0)
    indices_joint = tf.cast(tf.expand_dims(tf.tile(tf.range(num_keypoints), [batch_size]), 1), dtype=tf.float32)
    indices_joint = tf.concat([indices_joint] * d, axis=0)

    #TODO: want to use some kind of range..
    indices_lt = tf.concat([tf.expand_dims(y_floor, 1), tf.expand_dims(x_floor, 1)], axis=1)
    indices_lc = tf.concat([tf.expand_dims(y_floor + 1, 1), tf.expand_dims(x_floor, 1)], axis=1)
    indices_lb = tf.concat([tf.expand_dims(y_floor + 2, 1), tf.expand_dims(x_floor, 1)], axis=1)
    indices_mt = tf.concat([tf.expand_dims(y_floor, 1), tf.expand_dims(x_floor + 1, 1)], axis=1)
    indices_mc = tf.concat([tf.expand_dims(y_floor + 1, 1), tf.expand_dims(x_floor + 1, 1)], axis=1)
    indices_mb = tf.concat([tf.expand_dims(y_floor + 2, 1), tf.expand_dims(x_floor + 1, 1)], axis=1)
    indices_rt = tf.concat([tf.expand_dims(y_floor, 1), tf.expand_dims(x_floor + 2, 1)], axis=1)
    indices_rc = tf.concat([tf.expand_dims(y_floor + 1, 1), tf.expand_dims(x_floor + 2, 1)], axis=1)
    indices_rb = tf.concat([tf.expand_dims(y_floor + 2, 1), tf.expand_dims(x_floor + 2, 1)], axis=1)

    indices = tf.concat([indices_lt, indices_lc, indices_lb, indices_mt, indices_mc, indices_mb, indices_rt, indices_rc,
                         indices_rb], axis=0)
    indices = tf.cast(tf.concat([tf.cast(indices_batch, dtype=tf.float32), tf.cast(indices, dtype=tf.float32),
                                 tf.cast(indices_joint, dtype=tf.float32)], axis=1), tf.int32)

    probs = tf.reshape(gaussian, (num_keypoints * batch_size * 9,))
    heatmap = tf.scatter_nd(indices, probs, (batch_size, *output_shape, num_keypoints))
    heatmap = tf.squeeze(heatmap)
    return heatmap

def tf_create_OH(coord, output_shape,num_keypoints):
    batch_size = 1
    input_shape = output_shape
    x = tf.reshape(coord[:, 0] / input_shape[1]*8 * output_shape[1]*8, [-1])
    y = tf.reshape(coord[:, 1] / input_shape[0]*8 * output_shape[0]*8, [-1])
    x_floor = tf.floor(x) - 1
    y_floor = tf.floor(y) - 1
    x_top = tf.floor(x) + 1
    y_top = tf.floor(y) + 1

    ones = tf.keras.backend.repeat(tf.eye(1), batch_size * num_keypoints)

    x_floor = tf.clip_by_value(x_floor, 3, output_shape[1]*8 - 3)  # fix out-of-bounds x
    y_floor = tf.clip_by_value(y_floor, 3, output_shape[0]*8 - 3)  # fix out-of-bounds y

    indices_batch = tf.expand_dims(tf.cast( \
        tf.reshape(
            tf.transpose( \
                tf.tile( \
                    tf.expand_dims(tf.range(batch_size), 0) \
                    , [num_keypoints, 1]) \
                , [1, 0]) \
            , [-1]), dtype=tf.float32), 1)
    indices_batch = tf.concat([indices_batch] * 1, axis=0)
    indices_joint = tf.cast(tf.expand_dims(tf.tile(tf.range(num_keypoints), [batch_size]), 1), dtype=tf.float32)
    indices_joint = tf.concat([indices_joint] * 1, axis=0)

    indices = tf.concat([tf.expand_dims(int((y_floor + 1)/8), 1), tf.expand_dims(int((x_floor + 1)/8), 1)], axis=1)
    indices = tf.cast(tf.concat([tf.cast(indices_batch, dtype=tf.float32), tf.cast(indices, dtype=tf.float32),
                                 tf.cast(indices_joint, dtype=tf.float32)], axis=1), tf.int32)


    probs = tf.reshape(ones, (num_keypoints * batch_size * 1,))
    heatmap = tf.scatter_nd(indices, probs, (batch_size, *output_shape, num_keypoints))
    heatmap = tf.squeeze(heatmap)
    return heatmap


def gen(num_samp, dir_path, data_set):
    #while True:
    xyz_list, K_list, indicies, num_samples = get_raw_data(dir_path, data_set)
    for i in range(num_samp):
        uv = projectPoints(xyz_list[i], K_list[i])
        z = np.array(xyz_list[i])[:,2]
        #print(z.shape)
        yield uv, z


def render_gaussian_heatmap(output_shape, sigma):
    x = [i for i in range(output_shape[1])]
    y = [i for i in range(output_shape[0])]
    xx, yy = tf.meshgrid(x, y)
    xx = tf.cast(xx, tf.float32)
    yy = tf.cast(yy, tf.float32)
    x = tf.floor(tf.constant(output_shape[0]/2.0, shape=output_shape))
    y = tf.floor(tf.constant(output_shape[1]/2.0, shape=output_shape))

    sigma = tf.cast(sigma, tf.float32)
    heatmap = tf.exp(-(((xx - x) / sigma) ** 2) / 2.0 - (((yy - y) / sigma) ** 2) / 2.0)

    return heatmap


def map_uv_to_hm(uv,z):
    """ Create a heatmap for each uv-coordinate """
    std = 2
    # gaussian = render_gaussian_heatmap((3,3), std) #TODO: This creates a gaussian
    num_kps = tf.shape(uv)[0]
    hm = tf_create_OH(uv, (28,28), num_kps)

    return hm,z


def benchmark(dataset, num_epochs=2):
    """ Measure the time it takes to process the data wo training """
    start_time = time.perf_counter()
    r = []
    for epoch_num in range(num_epochs):
        for sample in dataset:
            r.append(sample)
    tf.print("Execution time:", time.perf_counter() - start_time)

    return r


def tf_generator(dir_path, batch_size=8, num_samp = 100, data_set = 'training'):
    """ Create generator, right now seperate one for
        heatmaps and one to read images"""
    dataset_uv = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.int64, tf.float32),
        output_shapes=(tf.TensorShape([21, 2]), tf.TensorShape(21)),
        args=[num_samp, dir_path, data_set])
    dataset_hm = dataset_uv.map(map_uv_to_hm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_im = create_image_dataset(dir_path, num_samp, data_set)
    dataset = tf.data.Dataset.zip((dataset_im, dataset_hm))
    batched_dataset = dataset.repeat().batch(batch_size)
    return batched_dataset

