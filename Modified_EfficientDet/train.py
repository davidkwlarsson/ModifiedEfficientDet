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
from skimage.transform import resize 
import numpy as np
from PIL import Image





from tensorflow import keras
# import tensorflow.keras.backend as K
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator

from network import efficientdet
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

tf.compat.v1.disable_eager_execution()

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

def trainGenerator(dir_path):
    image_datagen = ImageDataGenerator(rescale = 1./255)
    heat_datagen = ImageDataGenerator(rescale = 1./255)
    batch_size = 32
    seed = 1
    image_generator = image_datagen.flow_from_directory(os.path.join(dir_path, 'training/rgb'),
                                        target_size = (224, 224, 3),
                                        class_mode = None,
                                        batch_size = batch_size,
                                        seed = seed)

    heat_generator = heat_datagen.flow_from_directory(os.path.join(dir_path, 'training/heatmaps'),
                                        target_size = (224, 224),
                                        class_mode = None,
                                        batch_size = batch_size,
                                        seed = seed)

    train_generator = zip(image_generator, heat_generator)
    for (img, heat) in train_generator:
        yield (img, heat)



def create_generators(dir_path):
    # create random transform generator for augmenting training data
    train_datagen = ImageDataGenerator(rescale = 1./255)
    
    # CREATE A NEW GENERAL DATAGENERATOR

    # val_datagen = ImageDataGenerator(rescale = 1./255)

    
    seed = 1
    
    image_generator = train_datagen.flow_from_directory(os.path.join(dir_path, 'training/rgb'),
                                        target_size = (224, 224, 3),
                                        class_mode = None,
                                        seed = seed)
    # validation_generator = val_datagen.flow_from_directory(os.path.join(dir_path,'evaluation\rgb'),
    #                                     target_size = (224,224,3),
    #                                     class_mode = None)

    coord_generator = train_datagen.flow_from_directory(os.path.join(dir_path, 'training/heatmaps'),
                                        target_size = (224, 224),
                                        class_mode = None,
                                        seed = seed)
    
    return image_generator # + coord_generator

def get_trainData(dir_path, num_samples = 100):
    print("Collecting data ... \n")
    imgs = []
    heats = []
    n = 0
    imgs_path = os.path.join(dir_path, 'training/rgb')
    for f in os.listdir(imgs_path):
        imgs.append(io.imread(os.path.join(imgs_path, f)))
        n += 1
        if n > num_samples:
            n = 0
            break

    heat_path = os.path.join(dir_path, 'training/heatmaps')
    for f in os.listdir(heat_path):
        heats.append(io.imread(os.path.join(heat_path, f)))
        n += 1
        if n > num_samples:
            n = 0
            break

    return np.array(imgs), np.array(heats)
    # image_names = glob.glob(os.path.join(dir_path, 'training/rgb/*.jpg'))
    # heatmaps_names = glob.glob(os.path.join(dir_path, 'training/heatmaps/*.jpg'))



def main():
    dir_path = sys.argv[1]
    phi = 0
    weighted_bifpn = True
    freeze_backbone = False
    tf.compat.v1.keras.backend.set_session(get_session())

    # create the generators
    # train_generator = trainGenerator(dir_path)
    images, heatmaps = get_trainData(dir_path)
    print("Number of images: %s and heatmaps: %s\n" %(len(images), len(heatmaps)))
    model = efficientdet(phi, weighted_bifpn=weighted_bifpn,
                            freeze_bn=freeze_backbone)

    
    # model_name = 'efficientnet-b{}'.format(phi)
    # file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
    # file_hash = WEIGHTS_HASHES[model_name][1]
    # weights_path = keras.utils.get_file(file_name,
    #                                     BASE_WEIGHTS_PATH + file_name,
    #                                     cache_subdir='models',
    #                                     file_hash=file_hash)
    # model.load_weights(weights_path, by_name=True)


    # freeze backbone layers
    if freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][phi]):
            model.layers[i].trainable = False

    # compile model
    print("Compiling model ... \n")
    model.compile(optimizer=Adam(lr=1e-3),
                    loss='binary_crossentropy')

    # print(model.summary())

    # start training
    # return model.fit_generator(
    #     generator=train_generator,
    #     steps_per_epoch=10,
    #     initial_epoch=0,
    #     epochs=10,
    #     verbose=1
        # validation_data=validation_generator
    # )
    model.fit(images, heatmaps, batch_size = 16, epochs = 10, verbose = 1)
    model.save('efficientdet.h5', save_format='tf')


if __name__ == '__main__':
    main()
