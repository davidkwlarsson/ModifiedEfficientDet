from generator import json_load
import os
import sys
import numpy as np
import tensorflow as tf

from generator import projectPoints
from help_functions import read_img
from testing_scatter import insert_gaussian, calculate_bbox, _central_crop, \
    create_gaussian


def render_onehot_heatmap(coord, output_shape, num_keypoints):
    batch_size = tf.shape(coord)[0]
    input_shape = output_shape

    x = tf.reshape(coord[:, :, 0] / input_shape[1] * output_shape[1], [-1])
    y = tf.reshape(coord[:, :, 1] / input_shape[0] * output_shape[0], [-1])
    x_floor = tf.floor(x)
    y_floor = tf.floor(y)

    x_floor = tf.clip_by_value(x_floor, 0, output_shape[1] - 1)  # fix out-of-bounds x
    y_floor = tf.clip_by_value(y_floor, 0, output_shape[0] - 1)  # fix out-of-bounds y

    indices_batch = tf.expand_dims(tf.to_float( \
        tf.reshape(
            tf.transpose( \
                tf.tile( \
                    tf.expand_dims(tf.range(batch_size), 0) \
                    , [num_keypoints, 1]) \
                , [1, 0]) \
            , [-1])), 1)
    indices_batch = tf.concat([indices_batch] * 4, axis=0)
    indices_joint = tf.to_float(tf.expand_dims(tf.tile(tf.range(num_keypoints), [batch_size]), 1))
    indices_joint = tf.concat([indices_joint] * 4, axis=0)

    indices_lt = tf.concat([tf.expand_dims(y_floor, 1), tf.expand_dims(x_floor, 1)], axis=1)
    indices_lb = tf.concat([tf.expand_dims(y_floor + 1, 1), tf.expand_dims(x_floor, 1)], axis=1)
    indices_rt = tf.concat([tf.expand_dims(y_floor, 1), tf.expand_dims(x_floor + 1, 1)], axis=1)
    indices_rb = tf.concat([tf.expand_dims(y_floor + 1, 1), tf.expand_dims(x_floor + 1, 1)], axis=1)
    indices = tf.concat([indices_lt, indices_lb, indices_rt, indices_rb], axis=0)
    indices = tf.cast(tf.concat([tf.to_float(indices_batch), tf.to_float(indices), tf.to_float(indices_joint)], axis=1),
                      tf.int32)

    prob_lt = (1 - (x - x_floor)) * (1 - (y - y_floor))
    prob_lb = (1 - (x - x_floor)) * (y - y_floor)
    prob_rt = (x - x_floor) * (1 - (y - y_floor))
    prob_rb = (x - x_floor) * (y - y_floor)
    probs = tf.concat([prob_lt, prob_lb, prob_rt, prob_rb], axis=0)
    #  probs = tf.concat([prob_lt,prob_lt,prob_lt,prob_lt], axis=0)

    heatmap = tf.scatter_nd(indices, probs, (batch_size, *output_shape, num_keypoints))
    normalizer = tf.reshape(tf.reduce_sum(heatmap, axis=[1, 2]), [batch_size, 1, 1, num_keypoints])
    normalizer = tf.where(tf.equal(normalizer, 0), tf.ones_like(normalizer), normalizer)
    heatmap = heatmap / normalizer

    return heatmap



def mydataGenerator(dir_path, batch_size=16, data_set='training'):
    # hm_all = np.array(read_csv(dir_path+'/hm.csv'))

    if data_set == 'training':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[:-560]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[:-560]
        K_list *= 4
        indicies = [i for i in range(32000)] + [i for i in range(32560, 64560)] + \
                   [i for i in range(65120, 97120)] + [i for i in range(97680, 129680)]
        print("Total number of training samples: ", num_samples, " and ", len(indicies))
    elif data_set == 'validation':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
        K_list *= 4
        indicies = [i for i in range(32000, 32560)] + [i for i in range(64560, 65120)] + \
                   [i for i in range(97120, 97680)] + [i for i in range(129680, 130240)]
        print("Total number of validation samples: ", num_samples, " and ", len(indicies))
    elif data_set == 'evaluation':
        xyz_list = json_load(os.path.join(dir_path, 'evaluation_xyz.json'))
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))

    else:
        print("No specified data found!")
        sys.exit()

    i = 0
    r = 7
    std = 2
    w = 224
    h = 224
    gaussian = create_gaussian(std, r)
    border = 20
    while True:
        batch_x = []
        # batch_y = [[], [], [], []]
        # batch_y = [[], [], []]
        batch_y = [[]]
        for j in range(batch_size):
            idx = indicies[i + j]
            img = read_img(idx, dir_path, 'training') / 255.0
            batch_x.append(img)
            uv = projectPoints(xyz_list[i], K_list[i])
            uv = uv[:, ::-1]
            box = calculate_bbox(uv, r)
            hm = []
            for b in box:
                hm.append(insert_gaussian(gaussian, bbox=b, image_shape=(w + border, h + border)))
            hm = np.transpose(hm, (1, 2, 0))
            hm = _central_crop(hm, 224, 224)
            with tf.Session() as sess:
                hm = sess.run(hm)
            batch_y[0].append(hm)
            if i + j == num_samples - 1:
                i = -j
        i += batch_size


        yield (np.array(batch_x), batch_y)