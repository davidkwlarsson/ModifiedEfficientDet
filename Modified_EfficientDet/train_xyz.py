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

from utils.fh_utils import *
from help_functions import *
from plot_functions import *
from tf_generator import tf_generator, benchmark
from tf_generator_depth import tf_generator_depth
from tf_generator_xyz import tf_generator_xyz
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from network_depth import efficientdet # , efficientdet_coord
from losses import *

from data_generators import *

tf.compat.v1.disable_eager_execution()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

def shape_target(xyz_list):
    xyz_new_list = []
    for i in range(len(xyz_list)):
        xyz_new = []
        n = 0
        for j in range(0,21*3,3):
            xyz_new.append([])
           # print('i, ', i)
           # print('j, ', j)
            xyz_new[n].append(np.array(xyz_list)[i][j])
            xyz_new[n].append(np.array(xyz_list)[i][j+1])
            xyz_new[n].append(np.array(xyz_list)[i][j+2])
            n += 1
        xyz_new_list.append(xyz_new)
    return np.array(xyz_new_list)

def main():
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
    if dataset == 'small_dataset':
        num_samp = 400
    else:
        num_samp = 128000
    batch_size = 8
    num_val_samp = 560
    #train_dataset = tf_generator(dir_path, 'training', batch_size=batch_size, num_samp=num_samp)

    traingen = tf_generator_xyz(dir_path, dataset, batch_size=batch_size, num_samp=num_samp)
    validgen = tf_generator_xyz(dir_path, 'validation', batch_size=batch_size, num_samp=num_val_samp)

    traingen = traingen.prefetch(batch_size)
    validgen = validgen.prefetch(batch_size)

    # print("Number of images: %s and heatmaps: %s\n" % (len(images), len(heatmaps)))
   # model = efficientdet(phi, weighted_bifpn=weighted_bifpn,
   #                      freeze_bn=freeze_backbone)
    # model = efficientdet_coord(phi, weighted_bifpn=weighted_bifpn,
    #
    #                       freeze_bn = freeze_backbone)
    depth = False
    only_depth = True
    if depth:
        losses = {"normalsize": weighted_bce, "xyz": 'mean_squared_error'}
        lossWeights = {"normalsize": 1, "xyz": 0}
    elif only_depth:
        losses = {"xyz": 'mean_squared_error'}#, "depth": 'mean_squared_error'}
        lossWeights = {"xyz": 1}
    else:
        losses = {"normalsize": weighted_bce}#, "depth": 'mean_squared_error'}
        lossWeights = {"normalsize": 1}

    # freeze backbone layers
   # if freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
   #     for i in range(1, [227, 329, 329, 374, 464, 566, 656][phi]):
    #        model.layers[i].trainable = False
    model = efficientdet(phi, weighted_bifpn=weighted_bifpn,
                         freeze_bn=freeze_backbone)
    model.load_weights('model.h5', by_name=True)
    # Freeze the hm layers
    for layer in model.layers[:-13]:
        layer.trainable = False
    # compile model
    print("Compiling model ... \n")
    model.compile(optimizer=Adam(lr=1e-3), loss=losses, loss_weights=lossWeights)
    print("Number of parameters in the model : ", model.count_params())
    print(model.summary())
    print(traingen)
    print(validgen)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                    ,steps_per_epoch = num_samp//batch_size, epochs = nbr_epochs, verbose=1, callbacks=[callback])
    plot_acc_loss(history)

    validgen2 = dataGenerator_xyz(dir_path, batch_size=20, data_set='validation')
    print(validgen2)
    (images, targets) = next(validgen2)
    print(np.shape(images))
    preds = model.predict(images)
    depth = True
    print(np.shape(preds))
    print(np.shape(targets))

    pred_xyz = shape_target(preds)#[0]
    true_xyz = targets[0]
    print('targets ', np.shape(targets))
    print('images ', np.shape(images))
    print('pred_xyz', np.shape(pred_xyz))
    print('true_xyz', np.shape(true_xyz))
  #  if(depth):
  #      pred_depth = preds[1]
  #      true_depth = targets[1]
  #      print('true_depth', np.shape(true_depth))
  #      print('pred_depth', np.shape(pred_depth))
    images = images[0]
    for i in range(10):
        save_coords(pred_xyz[i], images[i], 'pred_' + str(i))
        save_coords(true_xyz[i], images[i], 'target_' + str(i))
       # roots.append(np.array(xyz_preds[i])[0][2])


if __name__ == '__main__':
    tf.keras.backend.clear_session()
   # use_multiprocessing = True
    main()
