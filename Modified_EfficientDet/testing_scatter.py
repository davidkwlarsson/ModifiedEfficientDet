import os
import sys

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from generator import json_load
from heatmapsgen import projectPoints
from help_functions import read_img, plot_hm_with_images
#from train_s import get_session
def create_gaussian(std, radius):
    gaussian = np.zeros((radius * 2 + 1, radius * 2 + 1))
    xc, yc = (radius , radius )
    for x in range(radius * 2 + 1):
        for y in range(radius * 2 + 1):
            dist = math.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            if dist < radius:
                scale = 1  # otherwise predict only zeros
                gaussian[x][y] = scale * math.exp(-dist ** 2 / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))
    m = np.max(gaussian)
    for x in range(radius * 2 +1):
        for y in range(radius * 2 +1):
            gaussian[x][y] = gaussian[x][y] / m

    return np.ndarray.tolist(gaussian)


def _central_crop(image, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image: a 3-D image tensor
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    3-D tensor with cropped image.
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  amount_to_be_cropped_h = (height - crop_height)
  crop_top = amount_to_be_cropped_h // 2
  amount_to_be_cropped_w = (width - crop_width)
  crop_left = amount_to_be_cropped_w // 2
  return tf.slice(
      image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def insert_gaussian(gaussian, bbox, image_shape=(224,224,1)):
        """generate a 2-d binary mask, all zeros except the bbox area

          bbox: [xmin, ymin, xmax, ymax]
          image_shape: [height, width, depth], depth will be ignored

          return:
            2-d binary mask, zero-initialized
        """
      #  tf.compat.v1.keras.backend.set_session(get_session())


        with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
            num_keypoints = 2
            batch_size = 3
            xmin = tf.cast(tf.math.round(bbox[0]), dtype=tf.int32)
            ymin = tf.cast(tf.math.round(bbox[1]), dtype=tf.int32)
            xmax = tf.cast(tf.math.round(bbox[2]), dtype=tf.int32)
            ymax = tf.cast(tf.math.round(bbox[3]), dtype=tf.int32)
            indices_batch = tf.expand_dims(tf.to_float( \
                tf.reshape(
                    tf.transpose( \
                        tf.tile( \
                            tf.expand_dims(tf.range(batch_size), 0) \
                            , [num_keypoints, 1]) \
                        , [1, 0]) \
                    , [-1])), 1)
            indices_batch = tf.concat([indices_batch, indices_batch, indices_batch, indices_batch], axis=0)
            indices_joint = tf.to_float(tf.expand_dims(tf.tile(tf.range(num_keypoints), [batch_size]), 1))
            indices_joint = tf.concat([indices_joint, indices_joint, indices_joint, indices_joint], axis=0)
            r_range = tf.range(xmin, xmax)
            c_range = tf.range(ymin, ymax)
            i_coords, j_coords = tf.meshgrid(r_range, c_range)
            indices = tf.stack([j_coords, i_coords], axis=-1)
            indices = tf.reshape(indices, [tf.size(i_coords), 2])
          #  print(np.shape(gaussian)[0])
            with tf.Session() as sess:
                imgs = sess.run(indices)
                print(imgs)
            updates = tf.constant(np.reshape(gaussian,(np.shape(gaussian)[0]*np.shape(gaussian)[1])))
          #  updates = tf.ones(tf.size(i_coords), dtype=tf.int64)
           # print(updates)
            shape = (batch_size, *image_shape, num_keypoints)
            shape = [image_shape[0], image_shape[1]]
            mask = tf.scatter_nd(indices, updates, shape)
            output_tf = sess.run(mask)

        return output_tf

    # Test with NumPy calculation
    #output_np = np.zeros((3, 4, 2, 2))
    #output_np[np.arange(3), :, x, y] = bbox
    #print(np.all(output_tf == output_np))
    # True

def calculate_bbox(uv, radius):
    px = 20
    py = 20
    bbox = []
    for coord in uv:
        u = coord[1] + px
        v = coord[0] + py
        xc_im = np.round(u)
        yc_im = np.round(v)
        x_b = xc_im - radius
        x_t = xc_im + radius +1
        y_b = yc_im - radius
        y_t = yc_im + radius +1
        bbox.append(list((x_b,  y_b, x_t, y_t)))

    return bbox




if __name__ == '__main__':
    dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/" #sys.argv[1]
    num_samples = 2
    r = 7
    matplotlib.use('TkAgg')

    gaussian = create_gaussian(2, r)
    #plt.imshow(gaussian)

    #gaussian = [gaussian]*2
    #uv = [[10,20]
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    heatmaps = []
    uv = []
    box = []
    print('len of list: ', len(xyz_list))
    for i in range(2):# range(len(xyz_list)):
        uv_i = projectPoints(xyz_list[i], K_list[i])
        uv_i = uv_i[:, ::-1]
        uv.append(uv_i)
        b = calculate_bbox(uv_i, r)
        box.append(b)
    plt.show()
  #  print(box)
    w = 224
    h = 224
# print(np.shape(gaussian))
    all_hm = []
    for i in range(num_samples):
        hm = []
        for b in box[i]:
            tmp = insert_gaussian(gaussian, bbox=b, image_shape=(w+20,h+20))
            hm.append(tmp)
        hm = np.transpose(hm, (1, 2, 0))
        tmp = _central_crop(hm, 224, 224)
        with tf.Session() as sess:
            tmp = sess.run(tmp)
        all_hm.append(tmp)

    #print(np.shape(hm))
    images = []
    for i in range(num_samples):
        # load images
        img = read_img(i, dir_path, 'training')
        images.append(img)

    # test_2d_vec([gaussian])

    plt.imshow(all_hm[0][:,:,0])
    plt.show()
   # print(np.shape(all_hm))
    plot_hm_with_images(np.array(all_hm), np.array(all_hm), images, 0)



# rad i gaussian | column i gaussian | rad i ny matris | column i ny matris
# idx [[[0 0 0 1]
#   [0 1 0 1]
#   [0 2 0 1]
#   [0 3 0 1]]
#
#  [[1 0 0 1]
#   [1 1 0 1]
#   [1 2 0 1]
#   [1 3 0 1]]
#
#  [[2 0 1 1]
#   [2 1 1 1]
#   [2 2 1 1]
#   [2 3 1 1]]]