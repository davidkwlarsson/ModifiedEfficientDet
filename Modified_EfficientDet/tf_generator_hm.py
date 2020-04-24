import sys
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from heatmapsgen import projectPoints
from data_generators import get_raw_data
from plot_functions import *
from help_functions import *
def tf_render_heatmap(coord, input_shape, output_shape, num_keypoints, gaussian):
    batch_size = 1  # tf.shape(coord)[0]
    d = 9
   # print(np.shape(coord))
    gaussian = tf.keras.backend.repeat(gaussian, batch_size * num_keypoints)#, axis=1)
    #input_shape = output_shape
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


def augment(image,label):
    image = tf.image.random_contrast(image, 0.8, 2.0) # Random crop back to 28x28
    image = tf.image.random_brightness(image, max_delta=0.3) # Random brightness
    image = tf.image.random_saturation(image, 0.8, 2.0)
    return image,label


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


def create_image_dataset(dir_path, num_samples, dataset):
    image_path = ''
    if dataset == 'training':
        image_path = dir_path + 'training/rgb/*'
    elif dataset == 'validation':
        image_path = dir_path + 'training/validation/*'
    elif dataset == 'small_dataset':
        image_path = dir_path + 'training/small_dataset/*'
    print('image_path: ', image_path)
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


def reshape_target(xyz_list):
    xyz_new = []
    for xyz in xyz_list:
        xyz_new.append(xyz[0])
        xyz_new.append(xyz[1])
        xyz_new.append(xyz[2])
    return np.array(xyz_new)


def gen(num_samp, xyz_list, K_list, s_list):
    for i in range(num_samp):
        #xyz = np.array(xyz_list[i])
        xy = np.array(xyz_list[i])[:, 0:-1]/s_list[i]
        z = np.array(xyz_list[i])[:, 2]/s_list[i]
        z_rel = z-z[0]

       # K = np.array(K_list[i])
        #Kinv = tf.linalg.inv(K)
       # f = np.mean((K[0][0], K[1][1]))
       # z = z / f * 20
        xyz = np.hstack((xy, np.expand_dims(z_rel, axis=1)))
        xyz = np.ndarray.flatten(xyz)
        uv = projectPoints(xyz_list[i], K_list[i])
        #uv = np.ndarray.flatten(uv)

        yield xyz, uv



def map_uv_to_hm(xyz, uv):
    """ Create a heatmap for each uv-coordinate """
    std = 2
    gaussian = render_gaussian_heatmap((3,3), std) #TODO: Move so only do this once
    num_kps = tf.shape(uv)[0]
    hm = tf_render_heatmap(uv,  (224,224), (56,56), num_kps, gaussian)
   # z = z - z[0]
   # print(np.shape(z))
    return xyz, hm


def benchmark(dataset, num_epochs=2):
    """ Measure the time it takes to process the data wo training """
    start_time = time.perf_counter()
    r = []
    for epoch_num in range(num_epochs):
        print(epoch_num)
        i = 0
        for sample in dataset:
           # print(sample)
            r.append(sample)
            i += 1
            if i > 2:
                break
        break


    tf.print("Execution time:", time.perf_counter() - start_time)

    return r





def tf_generator_hm(dir_path, dataset,  batch_size=3, num_samp = 3):
    """ Create generator, right now seperate one for
        heatmaps and one to read images"""
    xyz_list, K_list, num_samples, s_list = get_raw_data(dir_path, dataset)

    dataset_label = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([21*3]), tf.TensorShape([21,2])),
        args=[num_samples, xyz_list, K_list, s_list])
    dataset_label = dataset_label.map(map_uv_to_hm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_im = create_image_dataset(dir_path, num_samp, dataset)
    dataset_out = tf.data.Dataset.zip((dataset_im, dataset_label))
    if dataset == "training":
        dataset_out = dataset_out.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_out = dataset_out.shuffle(num_samples, reshuffle_each_iteration=True)

    batched_dataset = dataset_out.repeat().batch(batch_size)
    return batched_dataset

if __name__ == '__main__':

    try:
        dir_path = sys.argv[1]
    except:
        dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"  #

    print(tf.__version__)
  #  b = tf.constant([[1, 2], [3, 4]])
#
  #  t = tf.keras.backend.repeat(
  #      b, n = 2
  #  )
  #  print(t)
    list_of_samples = os.path.join(dir_path, 'training/rgb/*')
    num_samp = 3
    batch_size = 3
    batched_dataset = tf_generator_xyz(dir_path, 'small_dataset',  batch_size=batch_size, num_samp = num_samp)
    dataset = batched_dataset.unbatch()
    print('Dataset: ', dataset)
    #print(list(dataset.take(1).as_numpy_iterator()))
    i = 0
    set = benchmark(dataset, num_epochs=1)
    sum_scatter_hand = True
    plot_image = True
    if sum_scatter_hand:
        for sample in dataset:
            sum = 0
            for i in range(21):
             #   print(np.shape(sample[1]))
                sum = sum + sample[1][0][:,:,i]
            print(sample[1][1])
            plt.imshow(sum)
            plt.colorbar()
            plt.savefig('sum_fig_2_'+str(i)+'.png')
            break
    i = 0
    if plot_image:
        for sample in dataset:
            sum = sample[0]
            plt.imshow(sum)
            plt.savefig('image' + str(i) + '.png')
            i = i + 1
            break

    for sample in dataset:
        coords = heatmaps_to_coord([sample[1][0]])
        print(coords)
        coords = np.array(np.reshape(coords[0],(21,2)))
        print(coords)
        print(sample[1][1].numpy())
        plt.imshow(sample[0])
        plt.savefig('image.png')
        print(np.concatenate((coords,np.array([sample[1][1].numpy()]).T), axis=1))
        save_coords(np.concatenate((coords,np.array([sample[1][1].numpy()]).T), axis=1), sample[0])
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

