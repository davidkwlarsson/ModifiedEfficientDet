#import keras
import argparse
from tensorflow.keras import Sequential, applications, Model
from tensorflow.keras import layers
#from keras.datasets import cifar10
import matplotlib.pyplot as plt
from scipy.stats import norm
#from tqdm import tqdm
import pickle
import os
import math


os.environ['KMP_DUPLICATE_LIB_OK']='True'
#import create_data
#import view_samples
import numpy as np
#from keras_preprocessing.image import ImageDataGenerator
from utils.fh_utils import *


def simplemodel():
    base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = Sequential()
    model.add(base_model)
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    model.add(layers.Conv2D(21, (1, 1), activation='sigmoid', padding='same'))
    # model.add(layers.Conv2D(21, (1,1), activation='linear', padding='same', data_format='channels_first'))
    # model.add(layers.Reshape((21, 224, 224)))

    #for cnn_block_layer in model.layers[0].layers:
    #    cnn_block_layer.trainable = False
    #model.layers[0].trainable = False

    #model.compile(optimizer='adam', loss='mse', metrics=['accuracy']) #RMSprop

    return model


def coordmodel():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(42))
    return model

