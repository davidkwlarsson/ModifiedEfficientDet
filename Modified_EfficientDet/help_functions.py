import sys

import numpy as np
import matplotlib.pyplot as plt
from utils.fh_utils import *
import skimage.io as io
import math
from generator import projectPoints, json_load, _assert_exist
import os


def plot_heatmaps_with_coords(images, heatmaps, coords):
    # Here I display heatmaps and coordinates to check that the heatmaps are correctly generated
    #get data
    #dir_path = sys.argv[1]
    #images, heatmaps, coords = get_trainData(dir_path,multi_dim=True)
    hm = heatmaps[0][:, :, 0]
    for i in range(1,21):
        hm += heatmaps[0][:, :, i]
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(images[0])
    plt.scatter(coords[0][0::2], coords[0][1::2], marker='o', s=2)
    plt.subplot(1, 2, 2)
    plt.imshow(hm)
    plt.scatter(coords[0][0::2], coords[0][1::2], marker='o', color='r', s=2)
    plt.colorbar()
    plt.show()
    print(coords[0][0:2])


def plot_acc_loss(history):
    # Plot acc and loss vs epochs
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version))
    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)


def heatmaps_to_coord(heatmaps):
    coords = [[] for x in range(len(heatmaps))]
    for j in range(len(heatmaps)):
        for i in range(21):
            ind = np.unravel_index(np.argmax(heatmaps[j][:, :, i], axis=None), heatmaps[j][:, :, i].shape)
            coords[j].append(ind[0])
            coords[j].append(ind[1])

    return np.array(coords)



def create_gaussian_hm(uv, w, h):
    # TODO: Hantera kantfallen också
    hm_list = list()
    im = np.zeros((w, h))
    for coord in uv:
        u = coord[1]
        v = coord[0]
        radius = 20
        hm_small = np.zeros((radius * 2 + 1, radius * 2 + 1))
        xc, yc = (radius + 1, radius + 1)
        for x in range(radius * 2 + 1):
            for y in range(radius * 2 + 1):
                dist = math.sqrt((x - xc) ** 2 + (y - yc) ** 2)
                if dist < 5:
                    std = 2
                    scale = 1  # otherwise predict only zeros
                    hm_small[x][y] = scale * math.exp(-dist ** 2 / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))
        # plt.imshow(hm_small)
        # plt.show()
        # print('here')
        try:
            xc_im = np.round(u)
            yc_im = np.round(v)
            im[int(xc_im - radius - 1):int(xc_im + radius), int(yc_im - 1 - radius):int(yc_im + radius)] = hm_small
        except:
            print('Gaussian hm failed')
            continue
        hm_list.append(im)
        im = np.zeros((w, h))
    # plt.imshow(im)
    # plt.show()
    return np.transpose(np.array(hm_list), (1, 2, 0))


def get_trainData(dir_path, num_samples, multi_dim = True):
    print("Collecting data ... \n")
    imgs = []
    heats = []
    coords = []
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    for i in range(num_samples):
        # load images
        img = read_img(i, dir_path, 'training')
        imgs.append(img)
        # project 3d coords and create heatmaps
        uv = projectPoints(xyz_list[i], K_list[i])
        heats.append(create_gaussian_hm(uv,224,224))
        coords.append([])
        for j,coord in enumerate(uv):
            # save coordinates
            coords[i].append(coord[0])
            coords[i].append(coord[1])

    return np.array(imgs), np.array(heats), np.array(coords)


def trainGenerator(dir_path):
    image_datagen = ImageDataGenerator(rescale = 1./255)
    heat_datagen = ImageDataGenerator(rescale = 1)#"./255)
    batch_size = 32
    seed = 1
    image_generator = image_datagen.flow_from_directory(os.path.join(dir_path, 'training/rgb'),
                                        target_size = (224, 224, 3),
                                        class_mode = None,
                                        batch_size = batch_size,
                                        seed = seed)

    heat_generator = heat_datagen.flow_from_directory(os.path.join(dir_path, 'training/heatmaps'),
                                        target_size = (224, 224),
                                        class_mode = None,
                                        batch_size = batch_size,
                                        seed = seed)

    train_generator = zip(image_generator, heat_generator)
    for (img, heat) in train_generator:
        yield (img, heat)

def save_preds(dir_path, predictions):
    #Function that saves away the predictions as jpg images.
    save_path = dir_path + "predictions/test"
    makedirs(save_path)
    print("Saving the predictions to " + save_path)
    predictions *= 255
    for i, pred in enumerate(predictions):
        name = save_path + str(i)+".jpg"
        io.imsave(name, pred.astype(np.uint8))


def plot_predicted_heatmaps(preds, heatmaps):
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 5
    n = 0
    for i in range(1,rows+1,2):
        fig.add_subplot(rows, columns, i)
        plt.imshow(preds[0][:, :, n])
        plt.colorbar()
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(heatmaps[0][:, :, n])
        plt.colorbar()
        n += 1
    plt.show()
    plt.savefig('heatmaps.png')



def plot_predicted_coordinates(images, coord_preds, coord):
    try:
        fig = plt.figure(figsize=(8, 8))
        columns = 5
        rows = 2
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(images[i - 1])
            # need to project onto image..
            plt.scatter(coord[i - 1][0::2], coord[i - 1][1::2], marker='o', s=2)
            plt.scatter(coord_preds[i - 1][0::2], coord_preds[i - 1][1::2], marker='x', s=2)
        plt.show()
        plt.savefig('scatter.png')
    except:
        print('Error in scatter plot')


def keypoint_connections():
    connection_dict = dict()
    for i in range(21):
        connection_dict[i] = list()
        if i in [5, 9, 13, 17]:
            connection_dict[i].append(0)
        elif i == 0:
            connection_dict[i].append(1)
            connection_dict[i].append(5)
            connection_dict[i].append(9)
            connection_dict[i].append(13)
            connection_dict[i].append(17)
        else:
            connection_dict[i].append(i-1)
        if i not in [0, 4, 8, 12, 16, 20]:
            connection_dict[i].append(i + 1)

    return connection_dict


import tensorflow as tf
def build_connections(features):
    x = tf.ones((224, 224, 21))

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)  # execute init_op
        # print the random values that we sample
        print(sess.run(features[1]))


from tensorflow.keras import layers
from tensorflow.compat.v1.keras import backend as K

class ConnectKeypointLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        super(ConnectKeypointLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.conv2 = tf.keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu')
        self.conv2a = list()
        self.conv2b = list()
        for i in range(21):
            #TODO: How about activation here?
            self.conv2a.append(tf.keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu'))
            self.conv2b.append(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))



    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = list()
        for i in range(21):
            self.W.append(self.add_weight(name='kernel',
                                     shape=[int(input_shape[-1]),
                                      self.output_dim],
                                      #initializer='uniform',
                                     trainable=True))
        print(input_shape[1])
        print(self.output_dim)
        self.built = True

    def call(self, input):
        f = list()
        fn = list()
        c_dict = keypoint_connections()
        for i in range(21):
            f.append(self.conv2(input))
        for i in range(21):
            c = list()
            c.append(f[i])

            for j in c_dict[i]:
                c.append(f[j])

            c = K.concatenate(tuple(c), axis=-1)
            h = self.conv2a[i](c)
            g = self.conv2b[i](h)
            l = K.dot(input, self.W[i])
            #print(l.shape)
            fn.append(K.sum(K.concatenate((f[i], l), axis=-1), axis=-1, keepdims=True))

        return K.concatenate(tuple(fn), axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
