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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

#tf.enable_eager_execution()

import json
import glob
import skimage.io as io
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
#import cv2
from utils.fh_utils import *
from help_functions import *
from tf_generator import tf_generator, benchmark
#from tensorflow import keras
# import tensorflow.keras.backend as K
#from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
#from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.losses import Reduction
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
# from keras.optimizers import adam

from network import efficientdet # , efficientdet_coord
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from generator import projectPoints, json_load, _assert_exist
from losses import *
# from create_data import create_gaussian_hm
# from simple_model import simplemodel, coordmodel
import time

from data_generators import *

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

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

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
   # init_gaussian()
    cont_training = False
    weighted_bifpn = True
    freeze_backbone = False
    tf.compat.v1.keras.backend.set_session(get_session())
    # tf.config.gpu.set_per_process_memory_growth(True)

    # images, heatmaps, heatmaps2,heatmaps3, coord = get_trainData(dir_path, 100, multi_dim=True)
    #get_session()

   # hms, imgs = get_training_data(dir_path)
   # traingen = testthisoneGenerator(imgs, hms, batch_size = 16)
   # validgen = testthisoneGenerator(imgs, hms, batch_size= 16)
    batch_size = 16
    nbr_epochs = 1
    num_samp = 128000
    num_val_samp = 100
    train_dataset = tf_generator(dir_path, batch_size=batch_size, num_samp=num_samp)
    valid_dataset = tf_generator(dir_path, batch_size=batch_size, num_samp=num_val_samp)
    traingen = train_dataset.prefetch(batch_size)
    validgen = valid_dataset.prefetch(batch_size)
    # Create an iterator over the dataset
    #traingen = tf.compat.v1.data.make_one_shot_iterator(traingen)
   # validgen = tf.compat.v1.data.make_one_shot_iterator(validgen)

    # Initialize the iterator
  #  traingen = traingen.prefetch(tf.data.experimental.AUTOTUNE)
  #  validgen = validgen.prefetch(tf.data.experimental.AUTOTUNE)

  #  traingen = iter(traingen)
   # next_element = train_iterator.get_next()
   # validgen = iter(traingen)
    #next_element = val_iterator.get_next()


    #traingen = mytfdataGenerator(dir_path, batch_size=16, data_set='training')
    #validgen = mytfdataGenerator(dir_path, batch_size=16, data_set='training')
  #  dataset = tf.data.Dataset.from_generator(
      # traingen, (tf.int64, tf.float64))

    #list(dataset.take(3).as_numpy_iterator())

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
    #losses = {"normalsize" : weighted_bce, "size2" : weighted_bce, 'size3':weighted_bce, 'depthmaps' : 'mean_squared_error'}
    losses = {"size3" : weighted_bce}#, 'multiply' : weighted_bce}
    #losses = {'size3':weighted_bce}#, 'multiply' : weighted_bce}
    # lossWeights = {"normalsize" : 1.0, "size2" : 1.0, 'size3' : 1.0}
    lossWeights = {"size3" : 1.0}#, 'multiply' : 1.0}
    #lossWeights = { 'size3' : 1.0}#, 'multiply' : 1.0}
    #lossWeights = {"normalsize" : 1.0, "size2" : 1.0, 'size3' : 1.0, 'depthmaps' : 1.0}
    # focalloss = SigmoidFocalCrossEntropy(reduction=Reduction.SUM_OVER_BATCH_SIZE)
    model.compile(optimizer = Adam(lr=1e-3),
                    loss = losses, loss_weights = lossWeights)
    # model.compile(optimizer='adam', merics=['accuracy'], loss=weighted_bce)
    # loss=tf.keras.losses.SigmoidFocalCrossEntropy())
    # loss=weighted_bce)
    # loss=tf.keras.losses.SigmoidFocalCrossEntropy())
    # loss=[focal_loss(gamma = 2, alpha = 0.25)])
    # loss = 'mean_absolute_error'
   # print(model.summary())
    print("Number of parameters in the model : ", model.count_params())
    # print(get_flops(model))

    # model.fit(images, {"normalsize" : heatmaps, "size2": heatmaps2, 'size3': heatmaps3},
                                # batch_size=16, epochs=100, verbose=1)
    # K.set_value(model.optimizer.learning_rate, 1e-5)
    # model.fit(images, heatmaps, batch_size = 16, epochs = 100, verbose = 1)
   # tic = time.perf_counter()

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                    ,steps_per_epoch = num_samp//batch_size, epochs = nbr_epochs, verbose=1, callbacks=[callback])
    plot_acc_loss(history)

    #toc = time.perf_counter()
    # model.save_weights('handposenet')
   # validgen2 = mytfdataGenerator(dir_path, batch_size=16, data_set = 'validation')
   # validgen2 = tf_generator(dir_path, batch_size=4)
    validgen2 = create_image_tensor(dir_path, 10)
    images = validgen2
    images = images.batch(4)
    preds3 = model.predict(images)
    ### EVALUATE RESULT (not done)!
    evaluate_result(dir_path, preds3)


if __name__ == '__main__':
    tf.keras.backend.clear_session()
   # use_multiprocessing = True
    main()
