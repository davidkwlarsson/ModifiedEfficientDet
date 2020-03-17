import sys
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from heatmapsgen import projectPoints
from data_generators import create_image_dataset
from data_generators import get_raw_data


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


def gen(num_samp, dir_path):
    #while True:
    xyz_list, K_list, indicies, num_samples = get_raw_data(dir_path, 'training')
    for i in range(num_samp):
        uv = projectPoints(xyz_list[i], K_list[i])
        yield uv


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


def map_uv_to_hm(uv):
    """ Create a heatmap for each uv-coordinate """
    std = 2
    gaussian = render_gaussian_heatmap((3,3), std) #TODO: This creates a gaussian
    num_kps = tf.shape(uv)[0]
    hm = tf_render_heatmap(uv, (224,224), num_kps, gaussian)

    return hm


def benchmark(dataset, num_epochs=2):
    """ Measure the time it takes to process the data wo training """
    start_time = time.perf_counter()
    r = []
    for epoch_num in range(num_epochs):
        for sample in dataset:
            r.append(sample)
    tf.print("Execution time:", time.perf_counter() - start_time)

    return r


def tf_generator(dir_path, batch_size=8, num_samp = 100):
    """ Create generator, right now seperate one for
        heatmaps and one to read images"""
    dataset_uv = tf.data.Dataset.from_generator(
        gen,
        output_types=tf.int64,
        output_shapes=tf.TensorShape([21, 2]),
        args=[num_samp, dir_path])
    dataset_hm = dataset_uv.map(map_uv_to_hm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_im = create_image_dataset(dir_path, num_samp)
    dataset = tf.data.Dataset.zip((dataset_im, dataset_hm))
    batched_dataset = dataset.repeat().batch(batch_size)
    return batched_dataset


if __name__ == '__main__':

    try:
        dir_path = sys.argv[1]
    except:
        dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"  #


    list_of_samples = os.path.join(dir_path, 'training/rgb/*')
    num_samp = 8000
    batch_size = 16
    dataset1 = tf.data.Dataset.from_generator(
        gen,
        output_types=tf.int64,
        output_shapes=tf.TensorShape([21, 2]),
        args=[num_samp, dir_path])

    print('dataset from gen')
    print(dataset1)
    # print(list(dataset1.take(3).as_numpy_iterator()))
    dataset2 = dataset1.map(map_uv_to_hm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    im_data = create_image_dataset(dir_path, num_samp)
    print('dataset after map')
    dataset = tf.data.Dataset.zip((im_data, dataset2))
    print(dataset)
    batched_dataset = dataset.batch(batch_size)
    print('batched')

    print(batched_dataset)

   # print(list(dataset.take(3).as_numpy_iterator()))
    i = 0
    set = benchmark(dataset, num_epochs=2)
    print('after set')
    sum_scatter_hand = True
    plot_image = True
    if sum_scatter_hand:
        for sample in set:
            sum = 0
            for i in range(21):
                print(np.shape(sample[1]))
                sum = sum + sample[1][:,:,i]
            plt.imshow(sum)
            plt.colorbar()
            plt.savefig('tf_sum_fig_2'+str(i)+'.png')
            break
    i = 0
    if plot_image:
        for sample in set:
            sum = sample[0]
            plt.imshow(sum)
            plt.colorbar()
            plt.savefig('tf_image' + str(i) + '.png')
            i = i + 1
            break
#list_ds = tf.data.Dataset.list_files(filename, shuffle=False)
   # tf.enable_eager_execution
   # tf.enable_v2_behavior()
   # tf.enable_eager_execution()
    #dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

  #  def generator(list_of_samples):
  #      for filename in list_of_samples:
  #          #print(filename.numpy())
  #          label = 1
  #          yield filename, label

  #  def parse_sample(filename, label):
  #      print(filename)
  #      for f in filename:
  #          tf.print(f)
  #      image = tf.io.decode_jpeg(filename)
     #print(len(xyz_list))
    #     # Target är i ditt fall heatmaps
    #     # Räkna ut mha tensorflow-funktioner
   #     return image, label
   # list_ds = tf.data.Dataset.list_files(list_of_samples, shuffle=False)
   # for f in list_ds.take(5):
   #     print(f.numpy())
    #ds = tf.data.Dataset.from_generator(lambda: generator(list_ds),(tf.string, tf.int32))
                                          # Måste ange typ och storlek av vad generatorn ger också, kan vara lite meckigt att få rätt på
   # ds = ds.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
   # print('done')
   # print(ds)

