"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from datetime import date
import os
import sys
import tensorflow as tf
import json
import glob
import itertools
import skimage.io as io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from tensorflow import keras
# from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.losses import Reduction
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy

sys.path.insert(1, '../')


from network import efficientdet # , efficientdet_coord
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from FreiHAND.freihand_utils import *
from losses import *


# tf.compat.v1.disable_eager_execution()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """
    Construct a modified tf session.
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)

def get_flops(model):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()


    with graph.as_default():
        with session.as_default():

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Optional: save printed results to file
            # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
            # opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops


def scheduler(epoch):
        if epoch < 1:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1*(-epoch))

def tf_generator(x,y):
    for x_b, y_b in zip(x,y):
        yield (x_b,y_b)

default_timeit_steps = 1000
BATCH_SIZE = 16

@tf.function
def tf_onehots(xyz, K):

    def tf_projectPoints(xyz, K): 
        """ Project 3D coordinates into image space. """
        uv = tf.transpose(tf.linalg.matmul(K, tf.transpose(xyz)))
        return uv[:, :2] / uv[:, -1:]
    
    def tf_create_onehot(uv,h,w):
        uv = uv[:,::-1]
        temp_im = np.zeros(shape = (w,h,21))
        # temp_im2 = tf.zeros(shape = (w*2,h*2,21))
        # temp_im3 = tf.zeros(shape = (w*4,h*4,21))
        j = 0
        print(uv)
        for coord in uv:
            try:
                temp_im[int(coord[0]/8), int(coord[1]/8),j] = 1
                # temp_im2[int(coord[0]/2), int(coord[1]/2),j] = 1
                # temp_im3[int(coord[0]), int(coord[1]),j] = 1
                j += 1
            except:
                print("\n Coordinates where out of range : " , coord[0], coord[1])
                j += 1
        return tf.convert_to_tensor(temp_im) #, temp_im2, temp_im3
   
    
    def tf_get_depth(xyz):
        depth = np.zeros(shape = 21)
        
        for j in range(21):
            print(xyz[j,2])
            depth[j] = xyz[j,2]
            # depth[j] = 1
    
        return tf.convert_to_tensor(depth)
    


    uv = tf_projectPoints(xyz, K)

    onehots = tf_create_onehot(uv,28,28)
    depth = tf_get_depth(xyz)
    return onehots, depth #[0], onehots[1], onehots[2]

def get_tfdata(dir_path):
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))#[:300]
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))#[:300]
    length = len(xyz_list)*4
    # xyz_list *= 4
    xyz_data = tf.data.Dataset.from_tensor_slices(xyz_list)
    K_data = tf.data.Dataset.from_tensor_slices(K_list)
    heatmaps_ds = tf.data.Dataset.zip((xyz_data, K_data))
    # heatmaps_ds = heatmaps_ds.as_numpy_iterator()

    heatmaps_ds = heatmaps_ds.map(lambda x,y: tf_onehots(x,y))
    heatmaps_ds = heatmaps_ds.repeat(4)
    # for heats in heatmaps_ds.take(1):
    #     print(heats.numpy())
    
    image_path = os.path.join(dir_path, 'training/rgb/*')
    list_ds = tf.data.Dataset.list_files(image_path, shuffle = False)
    # for f in list_ds.take(5):
    #     print(f.numpy())


    # list_ds = list_ds.take(length)

    def get_tfimage(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img,channels = 3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224,224])
        return img
    list_ds = list_ds.map(get_tfimage)

    # for image in list_ds.take(1):
    #     print("Image shape: ", image.numpy().shape)

    labeled_ds = tf.data.Dataset.zip((list_ds, heatmaps_ds))
    labeled_ds = labeled_ds.shuffle(buffer_size = length, reshuffle_each_iteration = True)
    # labeled_ds = labeled_ds.repeat()
    labeled_ds = labeled_ds.batch(16)
    return labeled_ds

def main():
    dir_path = sys.argv[1]
    phi = 0
    cont_training = False
    weighted_bifpn = True
    freeze_backbone = False
    input_shape = (224,224,3)
    tf.compat.v1.keras.backend.set_session(get_session())

    # images, heatmaps, heatmaps2,heatmaps3, coord = get_trainData(dir_path, 100, multi_dim=True)
    
    train_data = get_tfdata(dir_path)

    # check if it looks good
    # plot_heatmaps_with_coords(images, heatmaps, coord)

    # print("Number of images: %s and heatmaps: %s\n" % (len(images), len(heatmaps)))
    model = efficientdet(phi, input_shape = input_shape,weighted_bifpn=weighted_bifpn,
                         freeze_bn=freeze_backbone)
    # model = efficientdet_coord(phi, weighted_bifpn=weighted_bifpn,
    #                       freeze_bn = freeze_backbone)

    # freeze backbone layers
    if freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][phi]):
            model.layers[i].trainable = False

    # compile model
    print("Compiling model ... \n")
    # losses = {"normalsize" : weighted_bce, "size2" : weighted_bce, 'size3':weighted_bce}
    losses = {"normalsize" : weighted_bce, 'depth' : 'mean_squared_error'}
    # # losses = {"normalsize" : weighted_bce, "size2" : weighted_bce, 'size3':weighted_bce, 'depthmaps' : 'mean_squared_error'}
    # lossWeights = {"normalsize" : 1.0, "size2" : 1.0, 'size3' : 1.0}
    lossWeights = {"normalsize" : 1.0, 'depth' : 1.0}
    # lossWeights = {"normalsize" : 1.0, "size2" : 1.0, 'size3' : 1.0, 'depthmaps' : 1.0}
    # focalloss = SigmoidFocalCrossEntropy(reduction=Reduction.SUM_OVER_BATCH_SIZE)
    model.compile(optimizer = Adam(lr=1e-3),
                    loss = losses, loss_weights = lossWeights,
                    metrics = {'normalsize' : 'mse'})
    # model.compile(optimizer='adam', metrics=['accuracy'], loss=weighted_bce)
    # loss=tf.keras.losses.SigmoidFocalCrossEntropy())
    # loss=weighted_bce)
    # loss=tf.keras.losses.SigmoidFocalCrossEntropy())
    # loss=[focal_loss(gamma = 2, alpha = 0.25)])
    # loss = 'mean_absolute_error'
    # print(model.summary())
    print("Number of parameters in the model : " ,model.count_params())
    # print(get_flops(model))

    # model.fit(images, {"normalsize" : heatmaps, "size2": heatmaps2, 'size3': heatmaps3},
    #                             batch_size=16, epochs=100, verbose=1)
    # K.set_value(model.optimizer.learning_rate, 1e-5)
    # model.fit(images, heatmaps, batch_size = 16, epochs = 100, verbose = 1)

    # callbacks = [
    # keras.callbacks.ModelCheckpoint(
    #     filepath='mymodel_{epoch}',
    #     # Path where to save the model
    #     # The two parameters below mean that we will overwrite
    #     # the current checkpoint if and only if
    #     # the `val_loss` score has improved.
    #     save_best_only=True,
    #     monitor='val_loss',
    #     verbose=1)
    #     ]


    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(train_data, epochs = 15, verbose = 1, callbacks = [callback])
    # model.save_weights('handposenet')

    # layer = model.get_layer('connect_keypoint_layer')
    # weights = layer.get_weights()
    # for weight, values in zip(layer.weights,weights):
    #     if weight.name == 'connect_keypoint_layer/lambda:0':
    #         print(weight, "with the value = ", values)
    # print(layer[-1]) 

    # images = get_evalImages(dir_path, 10)
    validgen2 = dataGenerator(dir_path, batch_size= 10, data_set = 'validation')
    (images, targets) = next(validgen2)

    # (preds, preds2 ,preds3, depth_pred) = model.predict(images)
    preds, depth_pred = model.predict(images)
    # (preds) = model.predict(images)
    
    # (heatmaps, heatmaps2, heatmaps3, depth) = targets
    heatmaps, depth = targets

    # preds = np.squeeze(preds)
    # heatmaps = np.squeeze(heatmaps)
    # (preds, preds2 ,preds3) = targets
    
    try:
        plot_acc_loss(history)
    except:
        print('could not plot loss')

    
    # get coordinates from predictions
    coord_preds = heatmaps_to_coord(preds)
    coord = heatmaps_to_coord(heatmaps)
    # coord_upsamp = heatmaps_to_coord(preds2)

    plot_predicted_heatmaps(preds, heatmaps, images)
    # plot_predicted_heatmaps(preds2, heatmaps2)
    plot_predicted_hands_uv(images, coord_preds*8)


    K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
    K_list = K_list[:len(preds)]
    xyz_pred = add_depth_to_coords(coord_preds*8, depth_pred, K_list)
    save_coords(xyz_pred, images[0])
    # draw_3d_skeleton(xyz_pred, (224*2,224*2))
    plot_predicted_coordinates(images, coord_preds*8, coord*8)
    # plot_predicted_coordinates(images, coord_upsamp*2, coord)
    save_model(model)




if __name__ == '__main__':
    tf.keras.backend.clear_session()
    print("THIS IS USED FOR FREIHAND, MAKE SURE INPUT SIZE IN NETWORK IS CORRECT")
    main()
