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


from MobileNet.mobnetwork import efficientdet_mobnet # , efficientdet_coord
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from FreiHAND.freihand_utils import *
from FreiHAND.tfdatagen_frei import *
from losses import *


tf.compat.v1.disable_eager_execution()
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

def main():
    dir_path = sys.argv[1]
    phi = 0
    cont_training = False
    weighted_bifpn = True
    freeze_backbone = False
    train_full = False
    train_uv = True
    train_z = True
    use_saved_model = False
    input_shape = (224,224,3)
    tf.compat.v1.keras.backend.set_session(get_session())

    # images, heatmaps, heatmaps2,heatmaps3, coord = get_trainData(dir_path, 100, multi_dim=True)
    num_samp = 128000
    num_val_samp = 2240
    batch_size = 16
    train_dataset = tf_generator(dir_path, batch_size=batch_size, num_samp=num_samp, data_set = 'training')
    valid_dataset = tf_generator(dir_path, batch_size=batch_size, num_samp=num_val_samp, data_set = 'validation')
    traingen = train_dataset.prefetch(batch_size)
    validgen = valid_dataset.prefetch(batch_size)
    print(traingen)


    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='mobnet_check.h5',
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            save_best_only = True,
            save_weights_only= True,
            monitor='val_loss',
            verbose=1)


    # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True)

    lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=0)
    # print("Number of images: %s and heatmaps: %s\n" % (len(images), len(heatmaps)))
    model = efficientdet_mobnet(phi, input_shape = input_shape,weighted_bifpn=weighted_bifpn,
                         freeze_bn=freeze_backbone)

    if use_saved_model:
        model.load_weights('mobnet.h5' , by_name = True)

    losses = {"uv_coords" : 'mean_squared_error', 'xyz_loss' : 'mean_squared_error' } #, 'xyz_loss' : 'mean_squared_error'}

    if train_full:
        lossWeights = {"uv_coords" : 1.0, 'xyz_loss' : 1.0} #, "xyz_loss" : 0.0}
        model.compile(optimizer = Adam(lr=1e-3),
                        loss = losses, loss_weights = lossWeights,
                        )
        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        print("Number of parameters in the model : " ,model.count_params())
        print(model.summary())
        history = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                        ,steps_per_epoch = num_samp//batch_size, epochs = 1, verbose=1, callbacks = [checkpoint, callback])


    if train_uv:
        print('train uv')
        # Freeze the liftpose layers
        alpha = 1.0 # tf.keras.backend.variable(1.0)
        beta = 0.0   # tf.keras.backend.variable(1.0)
        lossWeights = {"uv_coords" : alpha, 'xyz_loss' : beta} #, "xyz_loss" : 0.0}
        for layer in model.layers[-16:]:
            layer.trainable = False

        model.compile(optimizer = Adam(lr=1e-3),
                        loss = losses, loss_weights = lossWeights,
                        )
        print("Number of parameters in the model : " ,model.count_params())

        history_uv = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                        ,steps_per_epoch = num_samp//batch_size, epochs = 30, verbose=1, callbacks=[checkpoint, earlystop, lr_plateau])

    if train_z:
        for layer in model.layers[:-16]:
            layer.trainable = False
        for layer in model.layers[-16:]:
            layer.trainable = True

        lossWeights = {"uv_coords" : 0.0, 'xyz_loss' : 1.0}
        model.compile(optimizer = Adam(lr=1e-3),
                        loss = losses, loss_weights = lossWeights,
                        )

        
        print("Number of parameters in the model : " ,model.count_params())
        history_xyz = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                        ,steps_per_epoch = num_samp//batch_size, epochs = 5, verbose=1, callbacks = [earlystop, lr_plateau])


    
    # save_model(model, mob_net = True)
    for layer in model.layers:
            layer.trainable = True
    model.save_weights('mobnet.h5')
    model.save("saved_mobnet/my_model")

    validgen2 = dataGenerator(dir_path, batch_size= 16, data_set = 'validation')
    images, targets = next(validgen2)


    preds, xyz_pred = model.predict(images, verbose = 1)
    preds = np.array(preds[:10])
    xyz_pred = np.array(xyz_pred[:10])


    uv_target, depth_target = targets
    coord = np.array(uv_target[:10])
    coord = np.reshape(coord, (10,42))
    depth_target = np.array(depth_target[:10])

    # print(np.shape(xyz_pred))
    print(np.shape(coord), np.shape(preds))

    
    try:
        if train_uv:
            plot_acc_loss(history_uv, uv_only = True)
        if train_z:
            plot_acc_loss(history_xyz, xyz_only = True)
        if train_full:
            plot_acc_loss(history)
    except:
        print('could not plot loss')

    # get coordinates from predictions
    # coord_preds = heatmaps_to_coord(preds)
    coord_preds = np.reshape((np.array(preds)+1)*112, (10,42))
    print('predicted : ', coord_preds[0])
    print('target : ', coord[0])
    # coord = heatmaps_to_coord(heatmaps)

    plot_predicted_hands_uv(images, coord_preds)


    K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
    K_list = K_list[:len(preds)]
    s_list = json_load(os.path.join(dir_path, 'training_scale.json'))[-560:]
    s_list = s_list[:len(preds)]
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:]
    xyz_pred = add_relative(xyz_pred, xyz_list, s_list)
    # xyz_pred = add_depth_to_coords(coord_preds, z_pred, K_list, s_list)
    xyz_pred = np.reshape(xyz_pred, (-1, 63))
    save_coords(xyz_pred, images[0])
    plot_predicted_coordinates(images, coord_preds, coord)






if __name__ == '__main__':
    tf.keras.backend.clear_session()
    print("THIS IS USED FOR FREIHAND, MAKE SURE INPUT SIZE IN NETWORK IS CORRECT")
    main()
