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
import getopt
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

from MobileNet.mobilenet import efficientdet_mobnet  # , efficientdet_coord
from utils.fh_utils import *
from utils.tf_generator import tf_generator
from utils.plot_functions import *
from utils.help_functions import *

from utils.losses import weighted_bce

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
        return 0.001 * tf.math.exp(0.1 * (-epoch))


def main():
    """Use: python3 train_ed.py -f PATH_TO_FREIHAND -e NBR_OF_EPOCHS -t training"""
    ####
    # For when running locally
    nbr_epochs = 1
    # try:
    #     dir_path = sys.argv[2]
    # except:
    #     dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"
    #     print('in except')
    ####
    # -f FILE_PATH -e EPOCHS -t training/small_dataset
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:e:", ["freihand_path=", "epochs="])
        print('args,', args)
        print('opts,', opts)
    except getopt.GetoptError:
        print('Require inputs -f <freihand_path> -e <epochs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-e':
            nbr_epochs = int(arg)
        elif opt == '-f':
            dir_path = str(arg)

    phi = 0

    weighted_bifpn = True
    freeze_backbone = False
    tf.compat.v1.keras.backend.set_session(get_session())
    tot_num_samp = 130240

    num_train_samp = int(tot_num_samp * 0.85)
    batch_size = 16
    num_val_samp = int(tot_num_samp * 0.1)
    num_test_samp = int(tot_num_samp * 0.05)

    # 2D is pretrained so hm-loss is okay and can be trained together with 3D
    full_train = True # if false only train hm part
    only_predict = False # Use existing weights
    include_bifpn = True

    # Only true to run against only xyz for the full training
    no_hm = True

    input_shape = (112,112,3)

    traingen = tf_generator(dir_path, 'training', batch_size=batch_size, full_train=full_train, im_size = input_shape, no_hm = no_hm)
    validgen = tf_generator(dir_path, 'validation', batch_size=batch_size, full_train=full_train, im_size = input_shape, no_hm = no_hm)

    traingen = traingen.prefetch(batch_size)
    validgen = validgen.prefetch(batch_size)

    print(traingen)

    # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

   # earlystop = tf.keras.callbacks.EarlyStopping(
   #     monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
   #     baseline=None, restore_best_weights=True)
   # print("Number of images: %s and heatmaps: %s\n" % (len(images), len(heatmaps)))
    model = efficientdet_mobnet(phi, input_shape=input_shape, weighted_bifpn=weighted_bifpn,
                                freeze_bn=freeze_backbone, full_train=full_train, include_bifpn = include_bifpn, no_hm = no_hm)
    if only_predict and full_train:
        model.load_weights('xyz_weights_mobnet_cp.h5', by_name=True)
    elif only_predict and not full_train:
        model.load_weights('hm_pre_train_mobnet.h5', by_name=True)
    elif full_train and not no_hm: #load pre-trained weights
        model.load_weights('hm_pre_train_mobnet.h5', by_name=True)

    if only_predict and full_train:
        losses = {"xyz": 'mse', 'hm': weighted_bce}
        lossWeights = {"xyz": 1, 'hm': 1}

        print("Compiling model ... \n")
        model.compile(optimizer=Adam(lr=1e-3),
                      loss=losses, loss_weights=lossWeights,
                      metrics = ['accuracy'],
                      )
    elif only_predict and not full_train:
        losses = {'hm': weighted_bce}
        lossWeights = {'hm': 1}

        print("Compiling model ... \n")
        model.compile(optimizer=Adam(lr=1e-3),
                      loss=losses, loss_weights=lossWeights,
                      metrics = ['accuracy'],
                      )
    elif full_train:
        if no_hm:
            losses = {"xyz": 'mse'}
            lossWeights = {"xyz": 1}
        else:
            losses = {"xyz": 'mse', 'hm': weighted_bce}
            lossWeights = {"xyz": 1, 'hm': 1}
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='xyz_weights_mobnet_cp.h5',
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            verbose=1)
        print("Compiling model ... \n")
        model.compile(optimizer=Adam(lr=1e-3),
                      loss=losses, loss_weights=lossWeights, metrics=['accuracy']
                      )
        callbacks =[checkpoint]
        print("Number of parameters in the model : ", model.count_params())
        # print(model.summary())
        history = model.fit(traingen, validation_data=validgen, validation_steps=num_val_samp // batch_size
                            , steps_per_epoch=num_train_samp // batch_size, epochs=nbr_epochs, verbose=1,
                            callbacks=callbacks)

        model.save_weights("xyz_weights_mobnet.h5")
        save_loss(history)
        model.save('saved_model/my_model')
    else:
        losses = {'hm': weighted_bce}  # , "depth": 'mean_squared_error'}
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='hm_pre_train_mobnet.h5',
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            verbose=1)

        print('train hm only')
        # Freeze the liftpose layers

        lossWeights = {'hm': 1}  # , "xyz_loss" : 0.0}

        model.compile(optimizer=Adam(lr=1e-3),
                      loss=losses, loss_weights=lossWeights, metrics=['accuracy']
                      )
        # print(model.summary())
        print('check so freeze is correct')
        print("Number of parameters in the model : ", model.count_params())
        callbacks =[checkpoint]

        history = model.fit(traingen, validation_data=validgen, validation_steps=num_val_samp // batch_size
                               ,steps_per_epoch=num_train_samp // batch_size, epochs=nbr_epochs, verbose=1,
                               callbacks=callbacks)
        model.save_weights("hm_weights_mobnet.h5")

        save_loss(history)
    
    save_result=True
    if save_result:
        validgen2 = tf_generator(dir_path, 'validation', batch_size=batch_size, full_train=full_train, im_size=input_shape, no_hm = no_hm)
        validgen3 = tf_generator(dir_path, 'test', batch_size=batch_size, full_train=full_train, im_size = input_shape, no_hm = no_hm)

        preds = model.predict(validgen2, steps=None)
        preds_test = model.predict(validgen3, steps=None)
        if full_train:
            np.savetxt('xyz_pred.csv', preds[0], delimiter=',')
            np.savetxt('xyz_pred_test.csv', preds_test[0], delimiter=',')
            if not no_hm:
                np.savetxt('hm_pred.csv', np.reshape(preds[1][0:100], (-1, 56 * 56)), delimiter=',')
                np.savetxt('uv_pred.csv', heatmaps_to_coord(preds[1]), delimiter=',')
                np.savetxt('hm_pred_test.csv', np.reshape(preds_test[1][0:100], (-1, 56 * 56)), delimiter=',')
                np.savetxt('uv_pred_test.csv', heatmaps_to_coord(preds_test[1]), delimiter=',')

        else:
            #np.savetxt('hm_pred_m.csv', np.reshape(preds[0:1000], (-1, 56 * 56)), delimiter=',')
            np.savetxt('uv_pred.csv', heatmaps_to_coord(preds), delimiter=',')
            np.savetxt('uv_pred_test.csv', heatmaps_to_coord(preds_test), delimiter=',')

    #TODO shuffle data
    # Run hetmap with only heatmap settings... never predicted


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    print("THIS IS USED FOR FREIHAND, MAKE SURE INPUT SIZE IN NETWORK IS CORRECT")
    main()