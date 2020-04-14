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
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.losses import Reduction
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy

sys.path.insert(1, '../')


from network import efficientdet # , efficientdet_coord
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from FreiHAND.freihand_utils import *
from FreiHAND.tfdatagen_frei import get_raw_data, create_image_dataset
from losses import *
from eval import EvalUtil, loadtxt

from FreiHAND.tfdatagen_frei import tf_generator, benchmark

from keypointconnector import spatial_soft_argmax

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model



tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


def gen(num_samp, dir_path, data_set):
    xyz_list, K_list, s_list,num_samples = get_raw_data(dir_path, data_set)
    for i in range(num_samp):
        uv = projectPoints(xyz_list[i], K_list[i])/112 - 1
        xy = np.array(xyz_list[i])[:, 0:-1]/s_list[i]
        z = np.array(xyz_list[i])[:, 2]/s_list[i]
        z = z - z[0]
        xyz = np.concatenate((xy, np.expand_dims(z, axis = -1)), axis = -1)
        xyz = np.ndarray.flatten(xyz)
        # tf.zeros_like(xyz_list[i])
        yield uv, xyz

def tf_generator(dir_path, batch_size=8, num_samp = 100, data_set = 'training'):
    """ Create generator, right now seperate one for
        heatmaps and one to read images"""
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32,tf.float32),
        output_shapes=(tf.TensorShape([21,2]),tf.TensorShape([63])),
        args=[num_samp, dir_path, data_set])
    batched_dataset = dataset.repeat().batch(batch_size)
    return batched_dataset



def simple_liftpose(uv_coords):
    # Save the output layer and then pop it to remove
    uv_coords = layers.Flatten()(uv_coords)

    print(uv_coords)
    r0 = layers.Dense(64, activation = 'linear')(uv_coords)
    
    print('r0 : ', r0)
    r1 = r0
    for i in range(2):
        ## Residual block ##
        r1 = layers.BatchNormalization()(r1)
        r1 = layers.Dropout(0.1)(r1)
        r1 = tf.keras.activations.relu(r1)
        r1 = layers.Dense(64, activation='linear')(r1)
    added = layers.Add()([r0, r1])
    # print('added : ', added)

    print('added : ', added)
    
    depth = layers.Dense(63, activation='linear', name = 'xyz')(added)
    # rel_depth = layers.Reshape((21,3), name = 'uv_depth')(rel_depth)

    return depth



def main():
    dir_path = sys.argv[1]
    batch_size = 16
    uv_coords = Input(shape=(21,2))
    xyz = simple_liftpose(uv_coords)
    model = Model(inputs=[uv_coords], outputs=[xyz])
    
    num_samp = 128000
    num_val_samp = 640
    train_dataset = tf_generator(dir_path, batch_size=batch_size, num_samp=num_samp, data_set = 'training')
    valid_dataset = tf_generator(dir_path, batch_size=batch_size, num_samp=num_val_samp, data_set = 'validation')
    traingen = train_dataset.prefetch(batch_size)
    validgen = valid_dataset.prefetch(batch_size)
    print(traingen)

    model.compile(optimizer = Adam(lr=1e-3),
                        loss = 'mean_squared_error',
                        )
    history = model.fit(traingen, validation_data = validgen, validation_steps = num_val_samp//batch_size
                        ,steps_per_epoch = num_samp//batch_size, epochs = 20, verbose=1)

    validgen2 = dataGenerator(dir_path, batch_size= 16, data_set = 'validation')
    images, targets = next(validgen2)


    preds, xyz_pred = model.predict(images, verbose = 1)
    preds = np.array(preds[:10])
    xyz_pred = np.array(xyz_pred[:10])


    uv_target, depth_target = targets
    coord = np.array(uv_target[:10])
    coord = np.reshape(coord, (10,42))
    depth_target = np.array(depth_target[:10])

    # print(np.shape(xyz_pred))
    print(np.shape(coord), np.shape(preds))

    
    try:
        plot_acc_loss(history)
    except:
        print('could not plot loss')




    K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
    K_list = K_list[:len(preds)]
    s_list = json_load(os.path.join(dir_path, 'training_scale.json'))[-560:]
    s_list = s_list[:len(preds)]
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:]
    xyz_pred = add_relative(xyz_pred, xyz_list, s_list)
    # xyz_pred = add_depth_to_coords(coord_preds, z_pred, K_list, s_list)
    xyz_pred = np.reshape(xyz_pred, (-1, 63))
    save_coords(xyz_pred, images[0])
    xyz_pred = loadtxt('pose_cam_xyz.csv', delimiter=',') # Predictions
    xyz_pred = np.array(xyz_pred).reshape(-1,21,3)
    xyz = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:] # Target for validation data
        # Make sure they are equal in length
    xyz = np.array(xyz[:len(xyz_pred)]).reshape(-1,21,3)
    print(np.array(xyz_pred).shape, np.array(xyz).shape)
    eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
    for i in range(len(xyz_pred)):
        xyz_i = np.array(xyz[i])
        xyz_pred_i = np.array(xyz_pred[i])
        ## print(xyz_i.shape)
        # print(len(xyz_i.shape))
        eval_xyz.feed(
            xyz_i,
            np.ones(xyz_i.shape[0]),
            xyz_pred_i
        )
    # print(np.sum(xyz - xyz_pred, axis=0))
    print(np.sum(xyz-xyz_pred,axis = 0))
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))





if __name__ == '__main__':
    tf.keras.backend.clear_session()
    print("THIS IS USED FOR FREIHAND, MAKE SURE INPUT SIZE IN NETWORK IS CORRECT")
    main()
