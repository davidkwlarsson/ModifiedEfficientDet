from generator import json_load
import os
import sys
import numpy as np
import tensorflow as tf
import math

from generator import projectPoints
from help_functions import read_img, create_gaussian_hm
from old_files.testing_scatter import insert_gaussian, calculate_bbox, _central_crop, \
    create_gaussian
import matplotlib.pyplot as plt
from old_files.test_scatter_2 import render_heatmap

gaussian = np.zeros((12,12))
radius = 1


def create_gaussian_blob(std, radius):
    hm_small = np.zeros((radius * 2 + 1, radius * 2 + 1))
    xc, yc = (radius , radius )
    for x in range(radius * 2 + 1):
        for y in range(radius * 2 + 1):
            dist = math.sqrt((x - xc) ** 2 + (y - yc) ** 2)
          #  if dist < radius:
            scale = 1  # otherwise predict only zeros
            hm_small[x][y] = scale * math.exp(-dist ** 2 / (2 * std ** 2)) #/ (std * math.sqrt(2 * math.pi))
    m = np.max(hm_small)
    #TODO: Here I just removed the scaling part so normalizing should not be necessary?
   # for x in range(radius * 2 + 1):
    #    for y in range(radius * 2 + 1):
     #       hm_small[x][y] = hm_small[x][y] / m

    return hm_small


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

def init_gaussian():
    std = 2
    global gaussian
    gaussian = create_gaussian(std, radius)
   # print('GAUSSIAN')
  #  print(gaussian)

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


def get_uv_data(xyz_list, K_list, num_samples):
    uv_all = []
    for idx in range(num_samples):
        uv = projectPoints(xyz_list[idx], K_list[idx])
      #  uv = uv[:, ::-1]
        uv_all.append(uv)
#        z.append(np.array(xyz_list[idx])[:,2])
    return uv_all




def get_raw_data(dir_path, data_set='training'):
    try:
        dir_path = dir_path.decode('ascii')
        data_set = data_set.decode('ascii')
        print('dataset', data_set)
    except:
        print('string ok')
   # xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
   # xyz_list *= 4
   # K_list = json_load(os.path.join(dir_path, 'training_K.json'))
   # #K_list *= 4'

    if data_set == 'training':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[:-560]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[:-560]
        K_list *= 4
        #indicies = [i for i in range(32000)] + [i for i in range(32560,64560)] + [i for i in range(65120,97120)] + [i for i in range(97680,129680)]
        print("Total number of training samples: ", num_samples)
    elif data_set == 'small_dataset':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[:100]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[:100]
        K_list *= 4
        # indicies = [i for i in range(32000,32560)] + [i for i in range(64560,65120)] + [i for i in range(97120, 97680)] + [i for i in range(129680,130240)]
        print("Total number of small_daraset samples: ", num_samples)
    elif data_set == 'validation':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
        K_list *= 4
       # indicies = [i for i in range(32000,32560)] + [i for i in range(64560,65120)] + [i for i in range(97120, 97680)] + [i for i in range(129680,130240)]
        print("Total number of validation samples: ", num_samples)
    elif data_set == 'evaluation':
        xyz_list = json_load(os.path.join(dir_path, 'evaluation_xyz.json'))
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))

    else:
        print("No specified data found!")
        print('dir_path')
        sys.exit()


    return xyz_list, K_list, num_samples#, indicies, num_samples




def mytfdataGenerator(dir_path, batch_size=16, data_set='training'):

    i = 0
    output_shape = (224,224)
    num_keypoints = 21
    xyz_list, K_list, indicies, num_samples = get_raw_data(dir_path, 'training')
    gaussian_blob = create_gaussian_blob(2, radius)
    while True:

        batch_x = []
        batch_y = [[]]
        #batch_c = [[]]

        coords = np.zeros((batch_size, num_keypoints,2))
        for j in range(batch_size):
            idx = indicies[i + j]
            img = read_img(idx, dir_path, 'training') / 255.0
            uv = projectPoints(xyz_list[i + j], K_list[i + j])
            uv = np.array(uv) / 4
            coords[j,:,:] = uv


          #  hm = create_gaussian_hm(uv, radius, gaussian, 224,224)
          #  batch_c[0].append(hm)
            batch_x.append(img)
            if i + j == num_samples - 1:
                i = -j
        i += batch_size
        hm1 = render_heatmap(coords, [x//4 for x in output_shape], num_keypoints, gaussian_blob)
       # hm2 = render_heatmap(coords, [x//2 for x in output_shape], num_keypoints, gaussian)
        #hm3 = render_heatmap(coords, output_shape, num_keypoints, gaussian)


          #  imgs = sess.run(hm1)
       # batch_y[0] = hm3.eval(session=tf.compat.v1.Session())
      #  plt.imshow(batch_y[0][0][:,:,0])
      #  plt.savefig('batch_hm')
        batch_y[0] = hm1.eval(session=tf.compat.v1.Session())

      #  print(np.shape(batch_y[0]))
#tf.data.Dataset.from_tensor_slices(hm1).batch(batch_size)  # tf.reshape(hm1, [batch_size, num_keypoints, -1])
       # batch_y[1] = tf.data.Dataset.from_tensor_slices(hm2).batch(batch_size)  # tf.reshape(hm1, [batch_size, num_keypoints, -1])
       # batch_y[2] = tf.data.Dataset.from_tensor_slices(hm3).batch(batch_size)  # tf.reshape(hm1, [batch_size, num_keypoints, -1])
       # print(np.shape(batch_y))
        #print(np.shape(batch_c))
       # batch_y[0] = tf.reshape(tf.transpose(hm1,[0,3,1,2]), [batch_size, num_keypoints, -1])
       # batch_y[1] =tf.reshape(tf.transpose(hm2,[0,3,1,2]), [batch_size, num_keypoints, -1])
       # batch_y[2] =tf.reshape(tf.transpose(hm3,[0,3,1,2]), [batch_size, num_keypoints, -1])
      #  print(np.shape(batch_y[0]))
      #  print(np.shape(batch_y[1]))

        yield (np.array(batch_x), batch_y)


def dataGenerator(dir_path, batch_size = 16, data_set = 'training'):
   # hm_all = np.array(read_csv(dir_path+'/hm.csv'))
    print(dir_path)
    r = 1
    gaussian = create_gaussian_blob(2,r)
    indicies = []
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
       # dir_path = dir_path + 'validation/'
    elif data_set == 'evaluation':
        xyz_list = json_load(os.path.join(dir_path, 'evaluation_xyz.json'))
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))

    else:
        print("No specified data found!")
        sys.exit()



    images = []
    hms = []

    for (i,idx) in enumerate(indicies):
        #print(data_set)
        img = read_img(idx, dir_path, data_set) / 255.0
        images.append(img)
        if i > 20:
            break
    for (i,(xyz, K)) in enumerate(zip(xyz_list, K_list)):
        uv = projectPoints(xyz, K)
        uv = uv[:, ::-1]
        hm = create_gaussian_hm(uv, 224, 224, r, gaussian)
        hms.append(hm[0])
        if i > 20:
            break
    yield ([images], [hms])


def dataGenerator_depth(dir_path, batch_size=16, data_set='training'):
    # hm_all = np.array(read_csv(dir_path+'/hm.csv'))
    print(dir_path)
    r = 1
    gaussian = create_gaussian_blob(2, r)
    indicies = []
    if data_set == 'training':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[:-560]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[:-560]
        K_list *= 4
        indicies = [i for i in range(32000)] + [i for i in range(32560, 64560)] + [i for i in range(65120, 97120)] + [i
                                                                                                                      for
                                                                                                                      i
                                                                                                                      in
                                                                                                                      range(
                                                                                                                          97680,
                                                                                                                          129680)]
        print("Total number of training samples: ", num_samples, " and ", len(indicies))

    elif data_set == 'validation':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
        K_list *= 4
        indicies = [i for i in range(32000, 32560)] + [i for i in range(64560, 65120)] + [i for i in
                                                                                          range(97120, 97680)] + [i for
                                                                                                                  i in
                                                                                                                  range(
                                                                                                                      129680,
                                                                                                                      130240)]
        print("Total number of validation samples: ", num_samples, " and ", len(indicies))
    # dir_path = dir_path + 'validation/'
    elif data_set == 'evaluation':
        xyz_list = json_load(os.path.join(dir_path, 'evaluation_xyz.json'))
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))

    else:
        print("No specified data found!")
        sys.exit()

    images = []
    hms = []
    depth = []
    for (i, idx) in enumerate(indicies):
        # print(data_set)
        img = read_img(idx, dir_path, data_set) / 255.0
        images.append(img)
        if i > 20:
            break

    for (i, (xyz, K)) in enumerate(zip(xyz_list, K_list)):
        uv = projectPoints(xyz, K)
      #  print(xyz)


        depth.append(np.array(xyz).T[2])
        uv = uv[:, ::-1]
        hm = create_gaussian_hm(uv, 224, 224, r, gaussian)
        hms.append(hm[0])
        if i > 20:
            break
    yield ([images], [hms, depth])

