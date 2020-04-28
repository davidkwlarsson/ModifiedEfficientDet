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
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from network import efficientdet # , efficientdet_coord
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
    traingen = tf_generator(dir_path, dataset, batch_size=batch_size, num_samp=num_samp)
    validgen = tf_generator(dir_path, 'validation', batch_size=batch_size, num_samp=num_val_samp)
   # traingen = train_dataset.prefetch(batch_size)
   # validgen = valid_dataset.prefetch(batch_size)

    # print("Number of images: %s and heatmaps: %s\n" % (len(images), len(heatmaps)))
    model = efficientdet(phi, weighted_bifpn=weighted_bifpn,
                         freeze_bn=freeze_backbone)
    # model = efficientdet_coord(phi, weighted_bifpn=weighted_bifpn,
    #
    #                       freeze_bn = freeze_backbone)
    depth = False
    if depth:
        losses = {"normalsize": weighted_bce, "depth": 'mean_squared_error'}
    else:
        losses = {"normalsize": weighted_bce}#, "depth": 'mean_squared_error'}
    # freeze backbone layers
    if freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][phi]):
            model.layers[i].trainable = False
    load = False
    if load:
        model.load_weights('model.h5', by_name=True)
        for layer in model.layers:
            layer.trainable = False
        model.compile(optimizer=Adam(lr=1e-3), loss=losses)

    else:
        # compile model
        print("Compiling model ... \n")
        model.compile(optimizer=Adam(lr=1e-3), loss=losses)
        print("Number of parameters in the model : ", model.count_params())
       # print(model.summary())

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        history = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                       ,steps_per_epoch = num_samp//batch_size, epochs = nbr_epochs, verbose=1, callbacks=[callback])
        model.save_weights("model.h5")

        plot_acc_loss(history)

    #toc = time.perf_counter()
    # model.save_weights('handposenet')
   # validgen2 = mytfdataGenerator(dir_path, batch_size=16, data_set = 'validation')
   # validgen2 = tf_generator(dir_path, batch_size=4)
  #  validgen2 = create_image_dataset(dir_path, 10)
  #  images = validgen2
  #  images = images.batch(10)
   # preds3 = model.predict(images)
  #  ### EVALUATE RESULT (not done)!
   # evaluate_result(dir_path, preds3)


    validgen2 = dataGenerator(dir_path, batch_size=20, data_set='validation')

    (images, targets) = next(validgen2)

    preds = model.predict(images)
    depth = False
    pred_hm = preds#[0]
    true_hm = targets[0]
    print('targets ', np.shape(targets))
    print('images ', np.shape(images))
    print('pred_hm', np.shape(pred_hm))
    print('true_hm', np.shape(true_hm))
    if(depth):
        pred_depth = preds[1]
        true_depth = targets[1]

        print('true_depth', np.shape(true_depth))
        print('pred_depth', np.shape(pred_depth))


    pred_coord = heatmaps_to_coord(pred_hm)
    true_coord = heatmaps_to_coord(true_hm)
    np.savetxt('uv_preds.csv',pred_coord,delimiter=',')
    np.savetxt('uv_targets.csv',true_coord,delimiter=',')
    images = images[0]
    plot_predicted_heatmaps(pred_hm, true_hm)

    # plot_predicted_depthmaps(depth)
    # print("Predicted first depthmap")
    # print(depth_tarval[0])
    # print(depth_predval[0])


    # Skeleton plot
    plot_predicted_hands_uv(images, pred_coord*4, 'uv_hands_pred.png')
    plot_predicted_hands_uv(images, true_coord*4, 'uv_hands.png')

    # xyz_pred = add_depth_to_coords(coord_preds[0], depth[0])
    # draw_3d_skeleton(xyz_pred, (224*2,224*2))
    # print(coord)
    # Scatter plot
    plot_predicted_coordinates(images, pred_coord*4, true_coord*4)
    # plot_predicted_coordinates(images, coord_upsamp*2, coord)

    plot_hm_with_images(pred_hm, true_hm, images, 0, 4)
    plot_hm_with_images(pred_hm, true_hm, images, 1, 4)
    plot_hm_with_images(pred_hm, true_hm, images, 2, 4)
    plot_hm_with_images(pred_hm, true_hm, images, 3, 4)
    plot_hm_with_images(pred_hm, true_hm, images, 4, 4)

    n = 0
    for sample in pred_hm:
        #print(np.shape(sample))
        sum = 0
        for k in range(21):
            sum = sum + sample[:, :, k]
        plt.figure()
        plt.imshow(sum)
        plt.colorbar()
        plt.savefig('tf_sum_fig_preds_hm' + str(n) + '.png')
        plt.close()
        n = n + 1

    i = 0
    if depth:
        for i in range(10):
            coords = heatmaps_to_coord([pred_hm[i]])
            print('coords,' , np.shape(coords))
            coords = np.array(np.reshape(coords, (21, 2)))
            print(np.shape(coords))
            print('depth')
            print(np.shape(pred_depth[i]))
            print('pred_hm')
            print(np.shape(pred_hm[i]))
            plt.imshow(images[0])
            plt.savefig('image'+ str(i) +'.png')
           # print(np.concatenate((coords, np.array([sample[1]]).T), axis=1))
            save_coords(np.concatenate((coords, np.array([pred_depth[i]]).T), axis=1), images[i], 'from_train')



if __name__ == '__main__':
    tf.keras.backend.clear_session()
   # use_multiprocessing = True
    main()