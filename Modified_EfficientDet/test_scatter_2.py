import os
import sys

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from generator import json_load
from heatmapsgen import projectPoints
from help_functions import read_img, plot_hm_with_images
#from train_s import get_session
import tensorflow as tf
IMGSIZE = 10


def draw_lmarks(img, x):
    return tf.tensor_scatter_nd_update(img, x[0], x[1])

def test2(locations, vals ):
    img = tf.zeros((IMGSIZE,IMGSIZE), dtype="float32")
    x = [2, 3]
    draw_lmarks(img, x)
    imgs = tf.map_fn(draw_lmarks, [locations, vals], dtype="float32")
    return imgs


def test1():
    num_loc = 10
    im_dim = 32
    locations = tf.random.uniform((num_loc,2), maxval=im_dim, dtype=tf.int32)
    print(locations.shape)
    print(locations)

    centers = tf.scatter_nd(locations, [1]*num_loc, (im_dim, im_dim))
   # with tf.Session() as sess:
   #     locations = sess.run(locations)
   #     print(locations)

    return centers
   # heatmap  = tf.nn.conv2d(centers[None, :, :, None], heatmap_template[:, :, None, None], (1, 1, 1, 1), 'SAME')[0, :, :, 0]

def render_gaussian_heatmap(coord, output_shape, sigma, valid=None):
    x = [i for i in range(output_shape[1])]
    y = [i for i in range(output_shape[0])]
    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(tf.to_float(xx), (1, *output_shape, 1))
    yy = tf.reshape(tf.to_float(yy), (1, *output_shape, 1))

    x = tf.reshape(tf.to_float(coord[:, :, 0]), [-1, 1, 1, 2])
    y = tf.reshape((coord[:, :, 1]), [-1, 1, 1, 2])

    heatmap = tf.exp(-(((xx - tf.to_float(x)) / tf.to_float(sigma)) ** 2) / tf.to_float(2) -
                     (-(((yy - tf.to_float(y)) / tf.to_float(sigma)) ** 2) / tf.to_float(2)))

    if valid is not None:
        valid_mask = tf.reshape(valid, [-1, 1, 1, 2])
        heatmap = heatmap * valid_mask

    return heatmap * 255.





def render_heatmap(coord, output_shape,num_keypoints, gaussian):
    batch_size = tf.shape(coord)[0]
    d = 9
   # with tf.Session() as sess:
   #     b = sess.run(batch_size)
   # gaussian = tf.constant(gaussian, tf.float32)
    # gaussian = tf.ones((3,3))
    #print('Gaussian', gaussian)
   # c = tf.constant([b * num_keypoints, 1], tf.int32)
    #gaussian = tf.tile(gaussian, c)


    gaussian = tf.repeat(gaussian, batch_size*num_keypoints, axis=1)
    #with tf.Session() as sess:
    #    imgs = sess.run(gaussian)
    #    print(imgs)
# print(batch_size)
    input_shape = output_shape
    x = tf.reshape(coord[:, :, 0] / input_shape[1] * output_shape[1], [-1])
    y = tf.reshape(coord[:, :, 1] / input_shape[0] * output_shape[0], [-1])
  #  print(x)
    x_floor = tf.floor(x)-1
    y_floor = tf.floor(y)-1
    x_top = tf.floor(x)+1
    y_top = tf.floor(y)+1

    x_floor = tf.clip_by_value(x_floor, 0, output_shape[1] - 1)  # fix out-of-bounds x
    y_floor = tf.clip_by_value(y_floor, 0, output_shape[0] - 1)  # fix out-of-bounds y
    #TODO: fix top
    #x_floor = tf.clip_by_value(x_floor, 0, output_shape[1] - 1)  # fix out-of-bounds x
   # y_floor = tf.clip_by_value(y_floor, 0, output_shape[0] - 1)  # fix out-of-bounds y
    indices_batch = tf.expand_dims(tf.cast( \
        tf.reshape(
            tf.transpose( \
                tf.tile( \
                    tf.expand_dims(tf.range(batch_size), 0) \
                    , [num_keypoints, 1]) \
                , [1, 0]) \
            , [-1]), dtype=tf.float32), 1)
    indices_batch = tf.concat([indices_batch]*d, axis=0)
    indices_joint = tf.cast(tf.expand_dims(tf.tile(tf.range(num_keypoints), [batch_size]), 1), dtype=tf.float32)
    indices_joint = tf.concat([indices_joint]*d, axis=0)

    indices_lt = tf.concat([tf.expand_dims(y_floor, 1), tf.expand_dims(x_floor, 1)], axis=1)
    indices_lc = tf.concat([tf.expand_dims(y_floor+1, 1), tf.expand_dims(x_floor, 1)], axis=1)
    indices_lb = tf.concat([tf.expand_dims(y_floor + 2, 1), tf.expand_dims(x_floor, 1)], axis=1)
    indices_mt = tf.concat([tf.expand_dims(y_floor, 1), tf.expand_dims(x_floor+1, 1)], axis=1)
    indices_mc = tf.concat([tf.expand_dims(y_floor+1, 1), tf.expand_dims(x_floor+ 1, 1)], axis=1)
    indices_mb = tf.concat([tf.expand_dims(y_floor + 2, 1), tf.expand_dims(x_floor+ 1, 1)], axis=1)
    indices_rt = tf.concat([tf.expand_dims(y_floor, 1), tf.expand_dims(x_floor + 2, 1)], axis=1)
    indices_rc = tf.concat([tf.expand_dims(y_floor+1, 1), tf.expand_dims(x_floor+ 2, 1)], axis=1)
    indices_rb = tf.concat([tf.expand_dims(y_floor + 2, 1), tf.expand_dims(x_floor + 2, 1)], axis=1)
    indices = tf.concat([indices_lt, indices_lc,indices_lb,indices_mt,indices_mc, indices_mb, indices_rt,indices_rc,
                         indices_rb], axis=0)


   # indices = tf.reshape(indices, [tf.size(i_coords), 2])
   # indices = tf.concat([indices_lt, indices_lb, indices_rt, indices_rb], axis=0)

    indices = tf.cast(tf.concat([tf.cast(indices_batch, dtype=tf.float32), tf.cast(indices, dtype=tf.float32), tf.cast(indices_joint, dtype=tf.float32)], axis=1), tf.int32)
    # TODO: Repeat does not work...
   # probs = tf.concat(tf.constant(gaussian)*batch_size*num_keypoints), axis=0)
   # with tf.Session() as sess:
   #     imgs = sess.run(probs)
   # print(imgs)
    #probs = tf.concat([gaussian]*batch_size, axis=-1)
   # print('PROBS0', probs)
   # print('gaussian', gaussian)

    probs = tf.reshape(gaussian, (num_keypoints*batch_size*9,))
   # print('PROBS2', probs)
    #probs = tf.reshape(tf.ones((num_keypoints*batch_size,3,3)),(num_keypoints*batch_size*9,))
   # probs = tf.concat([prob_lt, prob_lc, prob_lb,prob_mt, prob_mc, prob_mb, prob_rt, prob_rc, prob_rb], axis=0)
   # probs = tf.concat([prob_lt,prob_lt,prob_lt,prob_lt], axis=0)
    #print(probs)
    #print('PROBS1', probs)

    #print('indices', indices)
    heatmap = tf.scatter_nd(indices, probs, (batch_size, *output_shape, num_keypoints))
    #print('hm', heatmap)

    #normalizer = tf.reshape(tf.reduce_sum(heatmap, axis=[1, 2]), [batch_size, 1, 1, num_keypoints])
   # normalizer = tf.where(tf.equal(normalizer, 0), tf.ones_like(normalizer), normalizer)
  #  heatmap = heatmap / normalizer
  #  with tf.Session() as sess:
   #     imgs = sess.run(probs)
   #     print(imgs)

    return heatmap


def create_gaussian(std, radius):
    gaussian = np.zeros((radius * 2 + 1, radius * 2 + 1))
    xc, yc = (radius , radius)
    for x in range(radius * 2 + 1):
        for y in range(radius * 2 + 1):
            dist = math.sqrt((x - xc) ** 2 + (y - yc) ** 2)
         #   if dist <= radius:
            scale = 1  # otherwise predict only zeros
            print(scale * math.exp(-dist ** 2 / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi)))
            gaussian[x][y] = scale * math.exp(-dist ** 2 / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))
    m = np.max(gaussian)
    for x in range(radius * 2 + 1):
        for y in range(radius * 2 + 1):
            gaussian[x][y] = gaussian[x][y] / m
    #print(gaussian)

    return gaussian#np.ndarray.tolist(gaussian)



if __name__ == '__main__':
 #
    try:
        dir_path =  sys.argv[1]
    except:
        dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"  #

   #matplotlib.use('TkAgg')
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    uv = []
    batch_size = 16
    num_keypoints = 21

    a = np.zeros((batch_size, num_keypoints, 2))

    for i in range(batch_size):# range(len(xyz_list)):
        uv_i = projectPoints(xyz_list[i], K_list[i])
        uv_i = uv_i[:, ::-1]
        uv_i = np.array(uv_i)/4
        uv.append(uv_i)
        a[i, :, :] = uv_i
   # print(a)
   # a = np.array([[[1,2],[5,6],[1,2]],[[5,6],[5,6],[1,2]],[[5,6],[5,6],[1,2]]])

    #imgs = render_gaussian_heatmap(a, [10,10], 1)
    output_shape = (56,56)
    gaussian = create_gaussian(2, 1)
    #tf.constant(tf.convert_to_tensor(create_gaussian(2, 1))) #std,r
    hm = render_heatmap(a, output_shape, num_keypoints, gaussian)
    #features = tf.constant([[1, 3], [2, 1], [3, 3]])  # ==> 3x2 tensor
   # labels = tf.constant(['A', 'B', 'A'])  # ==> 3x1 tensor
   # dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    #print(dataset)
   # print('here',gaussian)
    #print(hm)
    imgs = hm.numpy()#eval(session=tf.compat.v1.Session())

    print(imgs.shape)
    print(imgs[0].shape)

    plt.imshow(imgs[0,:,:,0])
    plt.colorbar()
    plt.show()
    plt.savefig('test_img')
    #features = tf.constant([[1, 3], [2, 1], [3, 3]])  # ==> 3x2 tensor
    features = tf.constant(hm)
    #labels = tf.constant(['A', 'B', 'A'])  # ==> 3x1 tensor
    dataset = tf.data.Dataset.from_tensor_slices(features).batch(16)

   # dataset = tf.data.Dataset.from_tensor_slices(hm)
    print(dataset)
    print(list(dataset.as_numpy_iterator()))




