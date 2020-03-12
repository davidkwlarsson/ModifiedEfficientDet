import sys

import numpy as np



from data_generators import get_raw_data, create_gaussian_blob

from generator import projectPoints
import matplotlib.pyplot as plt
import matplotlib
from help_functions import read_img
import cv2
import tensorflow as tf

def create_gaussian_hm(uv, radius, gaussian_blob, input_size = 224, heatmap_size=64):
    #py = radius + 10
    #px = radius + 10
    num_kps = 21
    heatmaps = np.zeros((heatmap_size, heatmap_size, num_kps), dtype=np.float32)
    heatmap_to_input_ratio = float(heatmap_size) / input_size
    for k in range(num_kps):
        u = uv[k][0]
        v = uv[k][1]
        u_s = int(np.round(u * heatmap_to_input_ratio))
        v_s = int(np.round(v * heatmap_to_input_ratio))
        # if joint is visible, not the best solution..
        if u_s-radius-1 >= 0 or u_s + radius < heatmap_size or v_s-radius-1 >= 0 or v_s + radius > heatmap_size:
            heatmaps[u_s-radius-1:u_s + radius, v_s-radius-1:v_s + radius, k] = gaussian_blob

    return heatmaps
IMG_WIDTH = 224
IMG_HEIGHT = 224

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

if __name__ == '__main__':
    #dir_path = sys.argv[1]
    dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"  #
    matplotlib.use('TkAgg')
    hm_size = 224
    imgs = []
    xyz_list, K_list, indicies, num_samples = get_raw_data(dir_path, 'training')
    w = 224
    h = 224
    radius = 5
    gaussian_blob = create_gaussian_blob(2,radius)
    uv_all = []
    hm = []


    for idx in range(1):
        img = read_img(idx, dir_path, 'training') / 255.0
        #imgs.append(img)
        uv = projectPoints(xyz_list[idx], K_list[idx])
        uv = uv[:, ::-1]

        hm.append(create_gaussian_hm(uv, radius, gaussian_blob, heatmap_size=hm_size))
        uv_all.append(uv)
  #  print(uv_all[0])
   # plt.imshow(np.sum(hm[:, :, :], axis=-1))
   # plt.show()
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 5
    n = 0
    for i in range(1, rows*2, 2):
        fig.add_subplot(rows, columns, i)
        plt.imshow(cv2.resize(imgs[0], (hm_size,hm_size)))
        plt.colorbar()
        fig.add_subplot(rows, columns, i+1)
        h = hm[0]
        plt.imshow(h[:, :, i-1])
       # plt.imshow(np.sum(h[:, :, :], axis=-1))
        plt.colorbar()
        n += 1

    plt.show()

