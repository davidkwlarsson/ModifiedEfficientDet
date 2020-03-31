from functools import reduce
from tensorflow.compat.v1.keras import backend as K

# from keras import layers
# from keras import initializers
# from keras import models
# from keras_ import EfficientNetB0, EfficientNetB1, EfficientNetB2
# from keras_ import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization
from initializers import PriorProbability
from keypointconnector import *
from custom_layers import SumLayer, NormLayer
import numpy as np
w_bifpns = [64, 88, 112, 160, 224, 288, 384]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]


def DepthwiseConvBlock(kernel_size, strides, name, freeze_bn=False):
    f1 = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=False, name='{}_dconv'.format(name))
    f2 = BatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def ConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = layers.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                       use_bias=False, name='{}_conv'.format(name))
    f2 = BatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def build_BiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            C3)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            C4)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            C5)
        # P6_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
        #     C5)
        # P7_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
        #     P6_in)
    else:
        P3_in, P4_in, P5_in = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            P3_in)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            P4_in)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            P5_in)
        # P6_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
        #     P6_in)
        # P7_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
        #     P7_in)

    # upsample
    # P7_U = layers.UpSampling2D()(P7_in)
    # P6_td = layers.Add()([P7_U, P6_in])
    # P6_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P6'.format(id))(P6_td)
    # P6_U = layers.UpSampling2D()(P6_in)
    # P5_td = layers.Add()([P6_U, P5_in])
    # P5_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P5'.format(id))(P5_td)
    P5_U = layers.UpSampling2D()(P5_in)
    P4_td = layers.Add()([P5_U, P4_in])
    P4_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P4'.format(id))(P4_td)
    P4_U = layers.UpSampling2D()(P4_td)
    P3_out = layers.Add()([P4_U, P3_in])
    P3_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P3'.format(id))(P3_out)
    # downsample
    P3_D = layers.MaxPooling2D(strides=(2, 2))(P3_out)
    P4_out = layers.Add()([P3_D, P4_td, P4_in])
    P4_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P4'.format(id))(P4_out)
    P4_D = layers.MaxPooling2D(strides=(2, 2))(P4_out)
    P5_out = layers.Add()([P4_D, P5_in])
    P5_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P5'.format(id))(P5_out)
    # P5_D = layers.MaxPooling2D(strides=(2, 2))(P5_out)
    # P6_out = layers.Add()([P5_D, P6_in])
    # P6_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P6'.format(id))(P6_out)
    # P6_D = layers.MaxPooling2D(strides=(2, 2))(P6_out)
    # P7_out = layers.Add()([P6_D, P7_in])
    # P7_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P7'.format(id))(P7_out)

    return P3_out, P4_out, P5_out#, P6_out , P7_out


def build_wBiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            C3)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            C4)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            C5)
        # P6_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
        #     C5)
        # P7_in = ConvBlock(num_channels, kernel_size=3, strides=2, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
        #     P6_in)
    else:
        P3_in, P4_in, P5_in = features
        P3_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P3'.format(id))(
            P3_in)
        P4_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P4'.format(id))(
            P4_in)
        P5_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P5'.format(id))(
            P5_in)
        # P6_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P6'.format(id))(
        #     P6_in)
        # P7_in = ConvBlock(num_channels, kernel_size=1, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_P7'.format(id))(
        #     P7_in)

    # upsample
    # P7_U = layers.UpSampling2D()(P7_in)
    # P6_td = wBiFPNAdd(name='w_bi_fpn_add' if id == 0 else f'w_bi_fpn_add_{8 * id}')([P7_U, P6_in])
    # P6_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P6'.format(id))(P6_td)
    # P6_U = layers.UpSampling2D()(P6_in)
    # P5_td = wBiFPNAdd(name=f'w_bi_fpn_add_{8 * id + 1}')([P6_U, P5_in])
    # P5_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P5'.format(id))(P5_td)
    P5_U = layers.UpSampling2D()(P5_in)
    P4_td = wBiFPNAdd(name=f'w_bi_fpn_add_{8 * id + 2}')([P5_U, P4_in])
    P4_td = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P4'.format(id))(P4_td)
    P4_U = layers.UpSampling2D()(P4_td)
    P3_out = wBiFPNAdd(name=f'w_bi_fpn_add_{8 * id + 3}')([P4_U, P3_in])
    P3_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_U_P3'.format(id))(P3_out)
    # downsample
    P3_D = layers.MaxPooling2D(strides=(2, 2))(P3_out)
    P4_out = wBiFPNAdd(name=f'w_bi_fpn_add_{8 * id + 4}')([P3_D, P4_td, P4_in])
    P4_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P4'.format(id))(P4_out)
    P4_D = layers.MaxPooling2D(strides=(2, 2))(P4_out)
    P5_out = wBiFPNAdd(name=f'w_bi_fpn_add_{8 * id + 5}')([P4_D, P5_in])
    P5_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P5'.format(id))(P5_out)
    P5_D = layers.MaxPooling2D(strides=(2, 2))(P5_out)
    # P6_out = wBiFPNAdd(name=f'w_bi_fpn_add_{8 * id + 6}')([P5_D, P6_in])
    # P6_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P6'.format(id))(P6_out)
    # P6_D = layers.MaxPooling2D(strides=(2, 2))(P6_out)
    # P7_out = wBiFPNAdd(name=f'w_bi_fpn_add_{8 * id + 7}')([P6_D, P7_in])
    # P7_out = DepthwiseConvBlock(kernel_size=3, strides=1, freeze_bn=freeze_bn, name='BiFPN_{}_D_P7'.format(id))(P7_out)

    return P3_out, P4_out, P5_out#, P6_out , P7_out


# def build_regress_head(width, depth, num_anchors=9):
#     options = {
#         'kernel_size': 3,
#         'strides': 1,
#         'padding': 'same',
#         # 'kernel_initializer': initializers.normal(mean=0.0, stddev=0.01, seed=None),
#         'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
#         'bias_initializer': 'zeros'
#     }
#
#     inputs = layers.Input(shape=(None, None, width))
#     outputs = inputs
#     for i in range(depth):
#         outputs = layers.Conv2D(
#             filters=width,
#             activation='relu',
#             **options
#         )(outputs)
#
#     outputs = layers.Conv2D(num_anchors * 4, **options)(outputs)
#     # (b, num_anchors_this_feature_map, 4)
#     outputs = layers.Reshape((-1, 4))(outputs)
#
#     return models.Model(inputs=inputs, outputs=outputs, name='box_head')


def spatial_soft_argmax(features, heigth, width, channels, image_coords):
    # Assume features is of size [N, H, W, C] (batch_size, height, width, channels)
    # Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
    # jointly over the image dimensions
    H = heigth
    W = width
    C = channels
    temp = 0.01
    features = tf.reshape(tf.transpose(features, [0, 3, 1, 2]), [-1, H * W])
    print(tf.math.argmax(features, axis=-1))
    # print(tf.math.reduce_max(features, axis = -1))
    softmax = tf.nn.softmax(features / temp, axis=-1)
    # print(softmax)
    print(tf.math.argmax(softmax, axis=-1))
    # Reshape and transpose back to original format.
    softmax = tf.transpose(tf.reshape(softmax, [-1, C, H, W]), [0, 2, 3, 1])
    # Assume that image_coords is a tensor of size [H, W, 2] representing the image
    # coordinates of each pixel.
    # Convert softmax to shape [N, H, W, C, 1]
    softmax = tf.expand_dims(softmax, -1)
    # Convert image coords to shape [H, W, 1, 2]
    # im_init = tf.initializers.RandomUniform(minval = 0, maxval = 1)
    # image_coords = tf.Variable(initial_value = np.ones((self.H,self.W,2)), trainable = True, dtype = tf.float32)
    # print(image_coords)
    image_coords = tf.expand_dims(image_coords, 2)
    image_coords = tf.expand_dims(image_coords, 0)
    # print(softmax*image_coords)
    # print(softmax)
    # Multiply (with broadcasting) and reduce over image dimensions to get the result
    # of shape [N, C, 2]
    # print(softmax)
    # print(image_coords)
    spatial_soft_argmax = tf.reduce_sum(softmax * image_coords, axis=[1, 2])

    # Flatten the images and get the argmax in flatten mode
    # features = tf.reshape(features, [-1, heigth*width, channels])
    # spatial_argmax = tf.argmax(features, axis = 1)
    # print(spatial_argmax, ' number 1')
    # # 21 tensors with the flattened argmax index
    # new_spatial = []
    # spatial_argmax = tf.unstack(spatial_argmax, axis = -1)
    # for s_argmax in spatial_argmax:
    #     new_spatial.append(tf.transpose(tf.unravel_index(s_argmax, (heigth,width))))
    # 21 tensors with shape None,2 at this point
    # print(spatial_soft_argmax)
    # Stack the 21 tensors at the last axis: None, 2, 21.
    # spatial_argmax = tf.stack(new_spatial, axis = -1)
    # spatial_argmax = tf.cast(spatial_argmax, dtype = tf.float32)
    return spatial_soft_argmax


def liftpose(uv_coords, output_shape):
    # Save the output layer and then pop it to remove
    uv_coords = layers.Flatten()(uv_coords)
    print(uv_coords)
    r0 = layers.Dense(400, activation='linear')(uv_coords)
    print('r0 : ', r0)
    r1 = r0
    for i in range(2):
        ## Residual block ##
        r1 = layers.BatchNormalization()(r1)
        r1 = layers.Dropout(0.2)(r1)
        r1 = tf.keras.activations.relu(r1)
        r1 = layers.Dense(400, activation='linear')(r1)

    added = layers.Add()([r0, r1])
    print('added : ', added)

    rel_depth = layers.Dense(output_shape, activation='linear', name='xyz')(added)
    # rel_depth = layers.Reshape((21,3), name = 'uv_depth')(rel_depth)

    return rel_depth


def efficientdet(phi, num_classes=20, weighted_bifpn=False, freeze_bn=False, score_threshold=0.01):
    assert phi in range(7)
    input_size = image_sizes[phi]
    # input_shape = (input_size, input_size, 3)
    input_shape = (224, 224, 3)
    image_input = layers.Input(input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = 2 + phi
    w_head = w_bifpn
    d_head = 3 + int(phi / 3)
    backbone_cls = backbones[phi]
    weights = phi
    # features = backbone_cls(include_top=False, input_shape=input_shape, weights=weights)(image_input)
    features = backbone_cls(input_tensor=image_input, freeze_bn=freeze_bn)
    # for feat in features:
    #     print(feat.shape)

    if weighted_bifpn:
        for i in range(d_bifpn):
            features = build_wBiFPN(features, w_bifpn, i, freeze_bn=freeze_bn)
    else:
        for i in range(d_bifpn):
            features = build_BiFPN(features, w_bifpn, i, freeze_bn=freeze_bn)

    # regress_head = build_regress_head(w_head, d_head)
    # print("shape of output from final BiFPN layer: ", features[0].shape, features[1].shape, features[2].shape)

    ## OUTPUT SIZE OF 28,28,64
    feature3 = features[0]
    #feature2 = features[1]
    #feature3 = ConnectKeypointLayer(64)(feature3)
    #depth = layers.Flatten()(feature3)
    #depth = layers.Dense(21, activation = 'linear', name = 'depth')(depth)
    #split = tf.split(feature3, 32, axis = -1)


    feature2 = layers.UpSampling2D()(feature3) # from 28 -> 56
    #feature2_cont = layers.Conv2D(21, kernel_size = 3, strides = 1, padding = "same", activation = 'relu')(feature2)
    feature2 = layers.Conv2D(21, kernel_size = 3, strides = 1, padding = "same", activation = 'relu')(feature2)
   # feature2 = layers.Dropout(0.2)(feature2)
    feature2 = layers.Conv2D(21, kernel_size = 1, strides = 1, padding = "same", activation = 'sigmoid', name = 'normalsize')(feature2)
    num_rows, num_cols = (56,56)

    # This should be of shape (Batch_size, 21, 2)
    # Contains the uv_coords for the keyjoints based on heatmaps
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
    image_coords = tf.stack([x_pos, y_pos], axis=-1)
    # print('image coordinates : ', image_coords)

    uv_coords = spatial_soft_argmax(feature2, num_rows, num_rows, 21, image_coords)

    # set true if only the final part is trained
    # if include_depth == True:
    output_shape = 21*3
    uv_depth = liftpose(uv_coords, output_shape)
    model = models.Model(inputs=[image_input], outputs=[uv_depth])
    #model = models.Model(inputs=[image_input], outputs=[feature2, depth])

    return model

