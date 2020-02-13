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
import skimage.io as io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils.fh_utils import *
from help_functions import *

from tensorflow import keras
# import tensorflow.keras.backend as K
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import adam

from network import efficientdet, efficientdet_coord
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from generator import projectPoints, json_load, _assert_exist
from losses import *
from create_data import create_gaussian_hm
from simple_model import simplemodel, coordmodel

tf.compat.v1.disable_eager_execution()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    dir_path = sys.argv[1]
    phi = 0
    cont_training = False
    weighted_bifpn = True
    freeze_backbone = False
    tf.compat.v1.keras.backend.set_session(get_session())

    images, heatmaps, coord = get_trainData(dir_path, 10, multi_dim=True)
    # check if it looks good
    # plot_heatmaps_with_coords(images, heatmaps, coord)

    print("Number of images: %s and heatmaps: %s\n" % (len(images), len(heatmaps)))
    model = efficientdet(phi, weighted_bifpn=weighted_bifpn,
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
    model.compile(optimizer='adam', metrics=['accuracy'], loss=weighted_bce)
    # loss=tf.keras.losses.SigmoidFocalCrossEntropy())
    # loss=weighted_bce)
    # loss=tf.keras.losses.SigmoidFocalCrossEntropy())
    # loss=[focal_loss(gamma = 2, alpha = 0.25)])
    # loss = 'mean_absolute_error'
    print(model.summary())

    model.fit(images, heatmaps, batch_size=16, epochs=1, verbose=1)

    # model.save_weights('efficientdet')
    preds = model.predict(images[0:11])
    # plot_acc_loss(history)

    # get coordinates from predictions
    coord_preds = heatmaps_to_coord(preds)
    plot_predicted_heatmaps(preds, heatmaps)
    plot_predicted_coordinates(images, coord_preds, coord)


if __name__ == '__main__':
    main()
