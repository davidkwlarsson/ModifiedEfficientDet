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
# from tensorflow.keras.losses import Reduction
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
# from keras.optimizers import adam

from network import efficientdet # , efficientdet_coord
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from generator import projectPoints, json_load, _assert_exist
from losses import *
# from create_data import create_gaussian_hm
# from simple_model import simplemodel, coordmodel

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



def main():
    dir_path = sys.argv[1]
    phi = 0
    cont_training = False
    weighted_bifpn = True
    freeze_backbone = False
    tf.compat.v1.keras.backend.set_session(get_session())
    # tf.config.gpu.set_per_process_memory_growth(True)

    # images, heatmaps, heatmaps2,heatmaps3, coord = get_trainData(dir_path, 100, multi_dim=True)

    traingen = dataGenerator(dir_path, batch_size = 16, data_set = 'training')
    validgen = dataGenerator(dir_path, batch_size=16, data_set = 'validation')

    # check if it looks good
    # plot_heatmaps_with_coords(images, heatmaps, coord)

    # print("Number of images: %s and heatmaps: %s\n" % (len(images), len(heatmaps)))
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
    # losses = {"normalsize" : weighted_bce, "size2" : weighted_bce, 'size3':weighted_bce}
    losses = {"normalsize" : weighted_bce, "size2" : weighted_bce, 'size3':weighted_bce, 'depth' : 'mean_squared_error'}
    # losses = {"normalsize" : weighted_bce, "size2" : weighted_bce, 'size3':weighted_bce, 'depthmaps' : 'mean_squared_error'}
    # lossWeights = {"normalsize" : 1.0, "size2" : 1.0, 'size3' : 1.0}
    lossWeights = {"normalsize" : 1.0, "size2" : 1.0, 'size3' : 1.0, 'depth' : 1.0}
    # lossWeights = {"normalsize" : 1.0, "size2" : 1.0, 'size3' : 1.0, 'depthmaps' : 1.0}
    # focalloss = SigmoidFocalCrossEntropy(reduction=Reduction.SUM_OVER_BATCH_SIZE)
    model.compile(optimizer = Adam(lr=1e-3),
                    loss = losses, loss_weights = lossWeights)
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
                                # batch_size=16, epochs=100, verbose=1)
    # K.set_value(model.optimizer.learning_rate, 1e-5)
    # model.fit(images, heatmaps, batch_size = 16, epochs = 100, verbose = 1)

    model.fit(traingen, validation_data = validgen, validation_steps = 18
                    ,steps_per_epoch = 100, epochs = 100)

    # model.save_weights('handposenet')

    images = get_evalImages(dir_path, 10)

    # (preds, preds2 ,preds3) = model.predict(images)
    (preds, preds2 ,preds3, depth) = model.predict(images)
    # plot_acc_loss(history)
    
    # get coordinates from predictions
    coord_preds = heatmaps_to_coord(preds)
    # coord_upsamp = heatmaps_to_coord(preds2)

    # plot_predicted_heatmaps(preds, heatmaps)
    # plot_predicted_heatmaps(preds2, heatmaps2)
    plot_predicted_hands_uv(images, coord_preds*4)

    xyz_pred = add_depth_to_coords(coord_preds[0], depth[0])
    draw_3d_skeleton(xyz_pred, (224*2,224*2))
    # plot_predicted_coordinates(images, coord_preds*4, coord)
    # plot_predicted_coordinates(images, coord_upsamp*2, coord)





if __name__ == '__main__':
    tf.keras.backend.clear_session()
    main()
