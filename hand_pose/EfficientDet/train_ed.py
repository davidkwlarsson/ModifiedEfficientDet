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
import numpy as np
sys.path.insert(1, '../')

from EfficientDet.network import efficientdet # , efficientdet_coord
from utils.fh_utils import *
from utils.help_functions import *
from utils.plot_functions import *
from utils.tf_generator import tf_generator
from utils.losses import *

from tensorflow.keras.optimizers import Adam, SGD


tf.compat.v1.disable_eager_execution()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'


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


def main():
    """Use: python3 train_ed.py -f PATH_TO_FREIHAND -e NBR_OF_EPOCHS -t training"""
    ####
    # For when running locally
    nbr_epochs = 1
    dataset = 'small_dataset'
    try:
        dir_path = sys.argv[2]
        #nbr_epochs = int(sys.argv[2])
        #dataset = str(sys.argv[3])

    except:
        dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"
       # nbr_epochs = 1
       # dataset = 'small_dataset'
        print('in except')
    ####
    # -f FILE_PATH -e EPOCHS -t training/small_dataset
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:e:t:", ["epochs=", "train_path="])
        print('args,', args)
        print('opts,', opts)
    except getopt.GetoptError:
        print('Require inputs -e <epochs> -t <training_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-e':
            nbr_epochs = int(arg)
        elif opt == '-t':
            dataset = str(arg)
        elif opt == '-f':
            dir_path = str(arg)

    phi = 0
    cont_training = False
    weighted_bifpn = True
    freeze_backbone = False
    tf.compat.v1.keras.backend.set_session(get_session())
    tot_num_samp = 130240
    if dataset == 'small_dataset':
        num_train_samp = 400
    else:
        num_train_samp = int(tot_num_samp*0.85)
    batch_size = 16
    num_val_samp =int(tot_num_samp*0.1)
    num_test_samp= int(tot_num_samp*0.05)

    # 2D is pretrained so hm-loss is okay and can be trained together with 3D
    full_train = True
    train_hm = False # 2D is pretrained

    traingen = tf_generator(dir_path, dataset, batch_size=batch_size, full_train=full_train)
    validgen = tf_generator(dir_path, 'validation', batch_size=batch_size, full_train=full_train)

    traingen = traingen.prefetch(batch_size)
    validgen = validgen.prefetch(batch_size)

    model = efficientdet(phi, batch_size, weighted_bifpn=weighted_bifpn,
                         freeze_bn=freeze_backbone, full_train=full_train)

    # freeze backbone layers FALSE NOW
    if freeze_backbone:
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][phi]):
            model.layers[i].trainable = False
    if full_train:
        model.load_weights('hm_weights.h5', by_name=True)

    train = True
    if train:
        if full_train:
            losses = {"xyz": 'mse', "hm": weighted_bce}
            lossWeights = {"xyz": 1, "hm": 1}
            callbacks = [tf.keras.callbacks.ModelCheckpoint(
                filepath='xyz_weights_cp.h5',
                # Path where to save the model
                # The two parameters below muv_predsean that we will overwrite
                # the current checkpoint if and only if
                # the `val_loss` score has improved.
                save_weights_only=True,
                save_best_only=True,
                monitor='val_loss',
                verbose=1)
                ]
            print("Compiling model ... \n")
            model.compile(optimizer=Adam(lr=1e-3), loss=losses, loss_weights=lossWeights)
            print("Number of parameters in the model : ", model.count_params())
            print(model.summary())
            history = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                            ,steps_per_epoch = num_train_samp//batch_size, epochs = nbr_epochs, verbose=1, callbacks=callbacks)


            model.save_weights("xyz_weights.h5")
        else: # if only train2D
            losses = {"hm": weighted_bce}
            callbacks = [tf.keras.callbacks.ModelCheckpoint(
                filepath='hm_weights_cp.h5',
                # Path where to save the model
                # The two parameters below muv_predsean that we will overwrite
                # the current checkpoint if and only if
                # the `val_loss` score has improved.
                save_weights_only=True,
                save_best_only=True,
                monitor='val_loss',
                verbose=1),
            ]
            print("Compiling model ... \n")
            model.compile(optimizer=Adam(lr=1e-3), loss=losses)
            print("Number of parameters in the model : ", model.count_params())
            print(model.summary())

            history = model.fit(traingen, validation_data=validgen, validation_steps=num_val_samp // batch_size
                                , steps_per_epoch=num_train_samp // batch_size, epochs=nbr_epochs, verbose=1,
                                callbacks=callbacks)
            model.save_weights("hm_weights.h5")
        save_loss(history)
        model.save('saved_model/my_model')
    else: # if not train
        if full_train:
            losses = {"xyz": 'mse', "hm": weighted_bce}
            model.load_weights("xyz_weights.h5")
        else:
            losses = {"hm": weighted_bce}
            model.load_weights("hm_weights.h5")

        model.compile(optimizer=Adam(lr=1e-3), loss=losses)

    save_result = True

    if save_result:
        validgen2 = tf_generator(dir_path, 'validation', batch_size=batch_size, full_train=full_train)
        preds = model.predict(validgen2, steps=None)

        np.savetxt('xyz_pred.csv', preds[0], delimiter=',')
        np.savetxt('hm_pred.csv', np.reshape(preds[1][0:1000], (-1,56*56)), delimiter=',')
        np.savetxt('uv_pred.csv', heatmaps_to_coord(preds[1]), delimiter=',')



if __name__ == '__main__':
    tf.keras.backend.clear_session()
   # use_multiprocessing = True
    main()
