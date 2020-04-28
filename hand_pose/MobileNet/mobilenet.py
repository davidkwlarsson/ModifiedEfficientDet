import sys
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.python.keras.backend import is_keras_tensor

#sys.path.insert(1, '../')

from EfficientDet.network import *
from utils import inject_tfkeras_modules


def efficientdet_mobnet(phi, input_shape=(224, 224, 3), num_classes=20, weighted_bifpn=False, freeze_bn=False, score_threshold=0.01, full_train=True):
    assert phi in range(7)
    #input_size = image_sizes[phi]
    # input_shape = (input_size, input_size, 3)
    # input_shape = (224, 224, 3)
    num_rows, num_cols = (224, 224)
    input_shape = input_shape
    image_input = layers.Input(shape=input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = 2 + phi
    w_head = w_bifpn
    d_head = 3 + int(phi / 3)
    # print(image_input)
    backbone_cls = MobileNetV2(input_shape=input_shape, include_top=False)(image_input)
    weights = phi
    # features = backbone_cls(include_top=False, input_shape=input_shape, weights=weights)(image_input)
    features = backbone_cls
    f_list = []
    f_list.append(features)
    for i in range(4):
        features = layers.UpSampling2D()(features)
        features = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(features)
        f_list.append(features)

    f_list.reverse()
    features = f_list
    print(features)
    for feat in features:
        print(feat.shape)

    if weighted_bifpn:
        for i in range(d_bifpn):
            features = build_wBiFPN(features, w_bifpn, i, freeze_bn=freeze_bn)
    else:
        for i in range(d_bifpn):
            features = build_BiFPN(features, w_bifpn, i, freeze_bn=freeze_bn)

    # regress_head = build_regress_head(w_head, d_head)
    print("shape of output from final BiFPN layer: ", features[0].shape, features[1].shape, features[2].shape)

    feature3 = features[0]  ## OUTPUT SIZE OF 28,28,64 for freihand
    # feature2 = features[1]

    feature2 = layers.UpSampling2D()(feature3)  # from 28 -> 56
    # feature2_cont = layers.Conv2D(21, kernel_size = 3, strides = 1, padding = "same", activation = 'relu')(feature2)
    feature2 = layers.Conv2D(21, kernel_size=3, strides=1, padding="same", activation='relu')(feature2)
    # feature2 = layers.Dropout(0.2)(feature2)
    hm = layers.Conv2D(21, kernel_size=1, strides=1, padding="same", activation='sigmoid', name='hm')(feature2)
    # feature2 = layers.UpSampling2D()(feature2) # from 56 -> 112
    # feature2 = layers.UpSampling2D()(feature2) # from 112 -> 224
    if full_train:
        output_shape = 21 * 3

        feat = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(feature3)
        feat = layers.Conv2D(8, kernel_size=1, strides=1, padding="same", activation='relu')(feat)
        feat = layers.Flatten()(feat)

        hm_small = layers.MaxPooling2D(strides=(2,2))(hm)
        hm_small = layers.Flatten()(hm_small)
        in_pose = layers.Concatenate()([feat, hm_small])
       # in_pose = feat
        xyz = liftpose(in_pose, output_shape)

        model = models.Model(inputs=[image_input], outputs=[xyz, hm])
    else:
        model = models.Model(inputs=[image_input], outputs=[hm])

    return model