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
from tf_generator_hm import tf_generator_hm
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from network_hm import efficientdet # , efficientdet_coord
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


def add_focal(xyz_list, K_list):
    xyz_new = []
    for i in range(len(xyz_list)):
        xy = np.array(xyz_list[i])[:, 0:-1]
        z = np.array(xyz_list[i])[:, 2]
        K = np.array(K_list[i])
        f = np.mean((K[0][0], K[1][1]))
        z = z * f /500
        xyz = np.hstack((xy, np.expand_dims(z, axis=1)))
        xyz_new.append(xyz)
    return np.array(xyz_new)


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

    traingen = tf_generator_hm(dir_path, dataset, batch_size=batch_size, num_samp=num_samp)
    validgen = tf_generator_hm(dir_path, 'validation', batch_size=batch_size, num_samp=num_val_samp)

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
    name= "normalsize"
    losses = {"xyz": 'mean_squared_error', name: weighted_bce}#, "depth": 'mean_squared_error'}
    lossWeights = {"xyz": 1, name: 1}
    model = efficientdet(phi, batch_size, weighted_bifpn=weighted_bifpn,
                         freeze_bn=freeze_backbone)
    # freeze backbone layers
    if freeze_backbone:
      #   227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][phi]):
            model.layers[i].trainable = False

    model.load_weights('model.h5', by_name=True)
    # Freeze the hm layers
   # for layer in model.layers[:-13]:
   #     layer.trainable = False
    # compile model
    print("Compiling model ... \n")
    model.compile(optimizer=Adam(lr=1e-3), loss=losses, loss_weights=lossWeights)
    print("Number of parameters in the model : ", model.count_params())
    #print(model.summary())
    print(traingen)
    print(validgen)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                    ,steps_per_epoch = num_samp//batch_size, epochs = nbr_epochs, verbose=1, callbacks=[callback])
    plot_acc_loss(history)

    validgen2 = dataGenerator_hm(dir_path, batch_size=10, data_set='validation')
    print(validgen2)
    (images, targets) = next(validgen2)
    print(np.shape(images))

    preds = model.predict(images)
    depth = True
    #print(np.shape(preds))
    #print(np.shape(targets))
    #xyz_list, K_list, num_samples = get_raw_data(dir_path, 'validation')

    pred_xyz = shape_target(preds[0])#[0]
    pred_hm = preds[1]
    true_xyz = targets[0]
    true_hm = targets[1]
    images = images[0]
    print('targets ', np.shape(targets[0]))
   # print('K ', np.shape(K))
    print('images ', np.shape(images))
    print('pred_xyz', np.shape(pred_xyz))
    print('true_xyz', np.shape(true_xyz))
    print('pred_hm', np.shape(pred_hm))
    print('true_hm', np.shape(true_hm))
    #uv_projs = preds[1]
   # print(np.shape(uv_projs))

    pred_coord = heatmaps_to_coord(pred_hm)
    true_coord = heatmaps_to_coord(true_hm)
    np.savetxt('uv_preds2.csv',pred_coord,delimiter=',')
    np.savetxt('uv_targets2.csv',true_coord,delimiter=',')
    plot_predicted_heatmaps(pred_hm, true_hm)
    # Skeleton plot
    plot_predicted_hands_uv(images, pred_coord * 4, 'uv_hands_pred.png')
    plot_predicted_hands_uv(images, true_coord * 4, 'uv_hands.png')
    # Scatter plot
    plot_predicted_coordinates(images, pred_coord*4, true_coord*4)
    plot_hm_with_images(pred_hm, true_hm, images, 0, 4)
    plot_hm_with_images(pred_hm, true_hm, images, 1, 4)
    plot_hm_with_images(pred_hm, true_hm, images, 2, 4)
    plot_hm_with_images(pred_hm, true_hm, images, 3, 4)
    plot_hm_with_images(pred_hm, true_hm, images, 4, 4)
    for i in range(batch_size):
        save_coords(pred_xyz[i], images[i], 'pred_' + str(i))
        save_coords(true_xyz[i], images[i], 'target_' + str(i))
       # tmp = np.reshape(uv_projs[i], (21,2))
        save_coords(pred_coord*4,images[i], 'uv_' + str(i))



if __name__ == '__main__':
    tf.keras.backend.clear_session()
   # use_multiprocessing = True
    main()
