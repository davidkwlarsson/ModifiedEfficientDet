from __future__ import absolute_import, division, print_function, unicode_literals

from train_s import *
import sys
from help_functions import *
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import tkinter

import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib



if __name__ == '__main__':
    dir_path = sys.argv[1]
    img, uv, hm, coords = get_data(dir_path, 19200, multi_dim=True)
    print(hm.shape)
    print(coords.shape)
    i = 1
    matplotlib.use('TkAgg')
    plt.subplot(2, 2, 1)
    plt.imshow(np.sum(hm[i][:, :, :], axis=-1), aspect="auto")
    plt.subplot(2, 2, 3)
    plt.imshow(img[i], aspect="auto")
    plt.subplot(2, 2, 2)
    plt.imshow(img[i], aspect="auto")
    print(coords[i])
    # print(coord[0])
    plt.scatter(coords[i][0::2], coords[i][1::2])
    plt.xlim((0, 224))
    plt.ylim((0, 224))
    plt.gca().invert_yaxis()
    plt.subplot(2, 2, 4)
    #plt.imshow(np.sum(oh[i][:, :, :], axis=-1), aspect="auto")
    plt.show()




def tmp():
    validgen2 = dataGenerator(dir_path, batch_size= 20, data_set='training')
    (images, targets) = next(validgen2)
    oh, hm1, hm2 = targets
    coord = heatmaps_to_coord(hm2)


    i = 2
    print(coord[i])
    # PLOT THE DATA
    #xy = np.array([coord[0][0::2], coord[0][1::2]]).T
    #hm, hm4, hm5 = create_gaussian_hm(coord, 224, 224)
    #oh = create_onehot(coord, 56, 56)

    matplotlib.use('TkAgg')
    plt.subplot(2, 2, 1)
    #plt.imshow(np.sum(hm5[:, :, :], axis=-1), aspect="auto")
    plt.imshow(np.sum(hm2[i][:, :, :], axis=-1), aspect="auto")
    plt.subplot(2, 2, 3)
    plt.imshow(images[i], aspect="auto")
    plt.subplot(2, 2, 2)
    plt.imshow(images[i], aspect="auto")

    #print(coord[0])
    plt.scatter(coord[i][0::2], coord[i][1::2])
    plt.xlim((0,224))
    plt.ylim((0,224))
    plt.gca().invert_yaxis()
    plt.subplot(2, 2, 4)
    plt.imshow(np.sum(oh[i][:, :, :], axis=-1), aspect="auto")
    plt.show()

    # PLOT COORDINATES FROM IMAGE
    plot_predicted_coordinates(images, coord, coord)

    # PLOT SKELETON ok
    plot_predicted_hands_uv(images, coord)

    #plot_predicted_heatmaps(heatmaps, hm2)