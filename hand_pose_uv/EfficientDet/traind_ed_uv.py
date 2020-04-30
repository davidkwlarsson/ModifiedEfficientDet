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
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD

sys.path.insert(1, '../')


from EfficientDet.network import efficientdet # , efficientdet_coord
from EfficientDet.efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
# from losses import *

from utils.help_functions import plot_acc_loss, plot_acc_loss_2D
from utils.tf_generator_uv import tf_generator, benchmark, tf_generator_2D



tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()
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
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))



def main():
    dir_path = sys.argv[1]
    phi = 0

    # Only set this to true if there has been a previous trained full model that you want to improve
    use_saved_model = False

    # Set this to True to train agains xyz coordinate, should have a pretrained 2D model.
    train_full = False

    weighted_bifpn = True
    freeze_backbone = False
    
    # Set the input to 224, 112 or 56
    input_shape = (224,224,3)
    im_size = (224,224)
    
    tf.compat.v1.keras.backend.set_session(get_session())
    print(tf.__version__)

    batch_size = 16
    nbr_epochs = 10
    num_samp = 110704
    num_val_samp = 13024
    if train_full:
        train_dataset = tf_generator(dir_path, im_size = im_size,batch_size=batch_size, num_samp=num_samp, data_set = 'training')
        valid_dataset = tf_generator(dir_path, im_size = im_size,batch_size=batch_size, num_samp=num_val_samp, data_set = 'validation')
    else:
        train_dataset = tf_generator_2D(dir_path, im_size = im_size,batch_size=batch_size, num_samp=num_samp, data_set = 'training')
        valid_dataset = tf_generator_2D(dir_path, im_size = im_size,batch_size=batch_size, num_samp=num_val_samp, data_set = 'validation')
    traingen = train_dataset.prefetch(batch_size)
    validgen = valid_dataset.prefetch(batch_size)
    print(traingen)

    model = efficientdet(phi, input_shape = input_shape,
                        include_depth= False,
                        weighted_bifpn=weighted_bifpn,
                        freeze_bn=freeze_backbone,
                        train_full = train_full)
    model_name = 'model_' + str(im_size[0]) + '.h5'

    if train_full:
        checkpoint_name = 'model_checkpoint_' + str(im_size[0]) + '.h5'
        if use_saved_model:
            print("Using weights to improve from previous runs")
            model.load_weights(model_name , by_name = True)
        else:
            print("Loading weights from pretraining on uv coordinates only")
            model.load_weights('model_2D_' + str(im_size[0]) + '.h5')
    else:
        checkpoint_name = 'model_checkpoint_2D_' + str(im_size[0]) + '.h5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_name,
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            save_best_only = True,
            save_weights_only= True,
            monitor='val_loss',
            verbose=1)


    # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # earlystop = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    #     baseline=None, restore_best_weights=True)

    # lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto',
    #     min_delta=0.0001, cooldown=0, min_lr=0)




    print("Compiling model ... \n")


    if train_full:
        print("Training the full network")
        losses = {"uv_coords" : 'mean_squared_error', 'xyz' : 'mean_squared_error' }
        lossWeights = {"uv_coords" : 0.0, 'xyz' : 1.0} #, "xyz_loss" : 0.0}
        model.compile(optimizer = Adam(lr=1e-3),
                        loss = losses, loss_weights = lossWeights,
                        )

        print("Number of parameters in the model : " ,model.count_params())
        # print(model.summary())
        history = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                        ,steps_per_epoch = num_samp//batch_size, epochs = 1, verbose=1, callbacks = [checkpoint])
        try:
            plot_acc_loss(history, im_size = im_size[0])
        except:
            print('could not plot loss')
    else:
        print('Pretraining the uv weights')
        # Freeze the liftpose layers
        losses = {"uv_coords" : 'mean_squared_error'}
        lossWeights = {"uv_coords" : 1.0}

        model.compile(optimizer = Adam(lr=1e-3),
                        loss = losses, loss_weights = lossWeights)

        print("Number of parameters in the model : " ,model.count_params())

        history_uv = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                        ,steps_per_epoch = num_samp//batch_size, epochs = 1, verbose=1, callbacks=[checkpoint])
        try:
            plot_acc_loss_2D(history_uv, im_size = im_size[0])
        except:
            print('could not plot loss')


    for layer in model.layers:
            layer.trainable = True

    model.save_weights(model_name)
    save_model_name = 'saved_model/my_model_' + str(im_size[0])
    model.save(save_model_name)
    
    if train_full:
        valid_dataset2 = tf_generator(dir_path, im_size = im_size,
                                batch_size=batch_size, num_samp=num_val_samp,
                                data_set = 'validation')
        validgen2 = valid_dataset2.prefetch(batch_size)

        preds, xyz_pred = model.predict(validgen2, verbose = 1)
        preds = np.array(preds)
        xyz_pred = np.array(xyz_pred)

        print(np.shape(preds), np.shape(xyz_pred))

        coord_preds = np.reshape((np.array(preds)+1)*112, (-1,42))
        xyz_pred = np.reshape(xyz_pred, (-1, 63))

        np.savetxt('pose_cam_xyz_pred_' + str(im_size[0]) + '.csv',xyz_pred, delimiter=',')
        np.savetxt('pose_cam_uv_pred_ ' + str(im_size[0]) + '.csv',coord_preds, delimiter=',')
    else:
        valid_dataset2 = tf_generator_2D(dir_path, im_size = im_size,
                            batch_size=batch_size, num_samp=num_val_samp,
                            data_set = 'validation')
        validgen2 = valid_dataset2.prefetch(batch_size)

        preds= model.predict(validgen2, verbose = 1)
        preds = np.array(preds) 

        print(np.shape(preds))

        coord_preds = np.reshape((np.array(preds)+1)*112, (-1,42))
        np.savetxt('pose_cam_uv_2D_pred_' + str(im_size[0]) + '.csv',coord_preds, delimiter=',')
    

if __name__ == '__main__':
    tf.keras.backend.clear_session()
    print("THIS IS USED FOR FREIHAND, MAKE SURE INPUT SIZE IN NETWORK IS CORRECT")
    main()
