import sys
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.python.keras.backend import is_keras_tensor
from tensorflow.keras import layers

sys.path.insert(1, '../')

from EfficienDet.network import *
from utils import inject_tfkeras_modules

def efficientdet_mobnet(phi,input_shape = (224,224,3), num_classes=20, weighted_bifpn=False, freeze_bn=False, score_threshold=0.01):
    
    assert phi in range(7)
    input_size = image_sizes[phi]
    # input_shape = (input_size, input_size, 3)
    # input_shape = (224, 224, 3)
    input_shape = input_shape
    image_input = layers.Input(shape = input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = 2 + phi  
    w_head = w_bifpn
    d_head = 3 + int(phi / 3)
    # print(image_input)
    backbone_cls = MobileNetV2(input_shape = input_shape, include_top = False)(image_input)
    weights = phi
    # features = backbone_cls(include_top=False, input_shape=input_shape, weights=weights)(image_input)
    features = backbone_cls
    f_list = []
    f_list.append(features)
    for i in range(4):
        features = layers.UpSampling2D()(features)
        features = layers.Conv2D(64,kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(features)
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
    
    feature3 = features[0] ## OUTPUT SIZE OF 28,28,64 for freihand
    # feature2 = features[1]
    
    feature2 = layers.UpSampling2D()(feature3) # from 28 -> 56
    feature2_cont = layers.Conv2D(21, kernel_size = 3, strides = 1, padding = "same", activation = 'relu')(feature2)
    feature2_cont = layers.Conv2D(kernel_size = 1, strides = 1, padding = "same", activation = 'linear')(feature2_cont)
    feature = feature2_cont

    # This should be of shape (Batch_size, 21, 2)
    # Contains the uv_coords for the keyjoints based on heatmaps
    num_rows, num_cols = (56,56)
    x_pos = np.empty((num_rows, num_cols), np.float32)
    y_pos = np.empty((num_rows, num_cols), np.float32)

    # Assign values to positions
    for i in range(num_rows):
      for j in range(num_cols):
        x_pos[i, j] = 2.0 * j / (num_cols - 1.0) - 1.0
        y_pos[i, j] = 2.0 * i / (num_rows - 1.0) - 1.0
        # x_pos[i,j] = j
        # y_pos[i,j] = i

    # x_pos = tf.reshape(x_pos, [num_rows * num_cols])
    # y_pos = tf.reshape(y_pos, [num_rows * num_cols])
    image_coords = tf.stack([x_pos,y_pos], axis = -1)
    # print('image coordinates : ', image_coords)

    uv_coords = spatial_soft_argmax(feature,num_rows, num_cols, 21, image_coords)
    # uv_coords = tf.compat.v1.contrib.layers.spatial_softmax(features, name = 'uv_coords')
    uv_coords_out = layers.Layer(name = 'uv_coords')(uv_coords)
    
    # set true if only the final part is trained
    # if include_depth == True:
    z_can = liftpose(uv_coords, feature3)

    # xyz = calc_xyz(z_can, uv_coords)

    model = models.Model(inputs=[image_input], outputs=[uv_coords_out, z_can])
    return model