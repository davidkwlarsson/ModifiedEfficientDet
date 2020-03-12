from __future__ import absolute_import, division, print_function, unicode_literals

from train_s import *
import sys
from help_functions import *
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from create_data import *
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib



#def tmp3():




def tmp2():
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




def tmp(dir_path):
    validgen2 = dataGenerator(dir_path, batch_size= 20, data_set='training')
    (images, targets) = next(validgen2)
    oh, hm1, hm2 = targets
    coord = heatmaps_to_coord(hm2)
    i = 2
    #print(coord[i])
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
    plt.scatter(coord[i][0], coord[i][1])
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

    # plot_predicted_heatmaps(heatmaps, hm2)

def tmp3(dir_path):
    print(np.array(targets[0]).shape)
    plot_hm_with_images(targets[0], targets[0], images, 0)
    # tmp(dir_path)
    # print(targets[1][1])
    c = []
    for j in range(16):
        c.append([])
        for i in targets[1][j]:
            c[j].append(i[0])
            c[j].append(i[1])
    plot_predicted_hands(images, np.array(c))
    plot_predicted_coordinates(images, c, c)



if __name__ == '__main__':
    dir_path = sys.argv[1]
   # matplotlib.use('TkAgg')
    num_samples = 16
    #validgen2 = dataGenerator_save_hm(dir_path,batch_size=num_samples, data_set='training')
   # validgen2 = dataGenerator_save_hm(dir_path,batch_size=2, data_set='training')

   # (images, targets) = next(validgen2)
    heatmaps = create_heatmaps(dir_path, num_samples)
    write_heatmaps(heatmaps, num_samples, (224,224), dir_path)

    traingen = dataGenerator(dir_path, batch_size=16, data_set='training')

    '''' DATASET '''
    dataset = tf.data.Dataset.from_generator(traingen, (tf.float32, tf.float32), output_shapes=None, args=None)

    dataset = dataset.batch(batch_size=10)
    dataset = dataset.repeat(count=2)

    for batch, (x, y) in enumerate(dataset):
        pass
    print("batch: ", batch)
    print("Data shape: ", x.shape, y.shape)


