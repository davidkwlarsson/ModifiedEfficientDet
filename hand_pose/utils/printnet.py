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
from MobileNet.mobilenet import efficientdet_mobnet
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
    phi = 0
    cont_training = False
    weighted_bifpn = True
    freeze_backbone = False
    tf.compat.v1.keras.backend.set_session(get_session())
    # 2D is pretrained so hm-loss is okay and can be trained together with 3D
    full_net = True
    full_train = full_net
    include_bifpn = True
    ED = True
    input_shape = (224,224,3)
    im_size = input_shape
    batch_size = 16


    if ED:
        model = efficientdet(phi,im_size, batch_size, weighted_bifpn=weighted_bifpn,
                         freeze_bn=freeze_backbone, full_train=full_train)
    else:
        model = efficientdet_mobnet(phi, input_shape=input_shape, weighted_bifpn=weighted_bifpn,
                                freeze_bn=freeze_backbone, full_train=full_train, include_bifpn = include_bifpn)
    if full_net:
        losses = {"xyz": 'mse', 'hm': weighted_bce}
        lossWeights = {"xyz": 1, 'hm': 1}
        print("Compiling model ... \n")
        model.compile(optimizer=Adam(lr=1e-3), loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
        print("Number of parameters in the model : ", model.count_params())
        # if not ED:
            # print(model.summary())
            # i = 1
            # for layer in model.layers:
            #     if i == 2:
            #         for la in layer.layers:
            #             print("Mob number : ", i, "layer :", la)
            #             i += 1
            #     else:    
            #         print("number : ", i, "layer : ", layer)
            #         i += 1

    else: # if only train2D
        losses = {"hm": weighted_bce}
        print("Compiling model ... \n")
        model.compile(optimizer=Adam(lr=1e-3), loss=losses, metrics=['accuracy'])
        print("Number of parameters in the model : ", model.count_params())
        print(model.summary())


if __name__ == '__main__':
    tf.keras.backend.clear_session()
   # use_multiprocessing = True
    main()
