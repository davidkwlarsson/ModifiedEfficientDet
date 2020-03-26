import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import backend as K

def keypoint_connections():
    connection_dict = dict()
    for i in range(21):
        connection_dict[i] = list()
        if i in [5, 9, 13, 17]:
            connection_dict[i].append(0)
        elif i == 0:
            connection_dict[i].append(1)
            connection_dict[i].append(5)
            connection_dict[i].append(9)
            connection_dict[i].append(13)
            connection_dict[i].append(17)
        else:
            connection_dict[i].append(i-1)
        if i not in [0, 4, 8, 12, 16, 20]:
            connection_dict[i].append(i + 1)

    return connection_dict



class ConnectKeypointLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        super(ConnectKeypointLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.conv2 = list()
        self.conv2a = list()
        self.conv2b = list()
        for i in range(21):
            #TODO: How about activation here?
            self.conv2a.append(tf.keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu'))
            self.conv2b.append(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
            self.conv2.append(tf.keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu'))


    def get_config(self):
        base_config = super(ConnectKeypointLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # self.W = list()
        # for i in range(21):
        self.W = self.add_weight(name='lambda',
                                    shape=[21,
                                    1],
                                    initializer='uniform',
                                    trainable=True)

        self.built = True

    def call(self, input):
        f = list()
        fn = list()
        c_dict = keypoint_connections()
        for i in range(21):
            f.append(self.conv2[i](input))
        for i in range(21):
            c = list()
            c.append(f[i])

            for j in c_dict[i]:
                c.append(f[j])

            c = K.concatenate(tuple(c), axis=-1)
            h = self.conv2a[i](c)
            g = self.conv2b[i](h)
            l = g * self.W[i]
            fn.append(K.sum(K.concatenate((f[i], l), axis=-1), axis=-1, keepdims=True))

        return K.concatenate(tuple(fn), axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



# class Softargmax(layers.layer):
#     def __init__(self):
#         super().__init__()

def softargmax(x, beta=1e10):
  x = tf.convert_to_tensor(x)
  x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
  return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

import numpy as np
class Spatial_softargmax(layers.Layer):

    def __init__(self, heigth,width,channels, **kwargs):
        super(Spatial_softargmax, self).__init__(**kwargs)
        self.H = heigth
        self.W = width
        self.C = channels
    
    def build(self, input_shape):
        self.image_coords = self.add_weight(name = 'image_coords',
                                shape = (self.H, self.W,2), initializer = 'uniform',
                                trainable = True)
        self.build = True

    def get_config(self):
        base_config = super(Spatial_softargmax, self).get_config()

    def call(self, input):
        # Assume features is of size [N, H, W, C] (batch_size, height, width, channels)
        # Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
        # jointly over the image dimensions
        features = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, self.H * self.W])
        softmax = tf.nn.softmax(features)
        # Reshape and transpose back to original format.
        softmax = tf.transpose(tf.reshape(softmax, [-1, self.C, self.H, self.W]), [0, 2, 3, 1])
        # Assume that image_coords is a tensor of size [H, W, 2] representing the image
        # coordinates of each pixel.
        # Convert softmax to shape [N, H, W, C, 1]
        softmax = tf.expand_dims(softmax, -1)
        # Convert image coords to shape [H, W, 1, 2]
        # im_init = tf.initializers.RandomUniform(minval = 0, maxval = 1)
        # image_coords = tf.Variable(initial_value = np.ones((self.H,self.W,2)), trainable = True, dtype = tf.float32)
        # print(image_coords)
        image_coords = tf.expand_dims(self.image_coords, 2)
        # Multiply (with broadcasting) and reduce over image dimensions to get the result
        # of shape [N, C, 2]
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
        print(spatial_soft_argmax)
        # Stack the 21 tensors at the last axis: None, 2, 21. 
        # spatial_argmax = tf.stack(new_spatial, axis = -1)
        # spatial_argmax = tf.cast(spatial_argmax, dtype = tf.float32)
        return spatial_soft_argmax/224


def spatial_soft_argmax(features,heigth,width,channels, image_coords):
    # Assume features is of size [N, H, W, C] (batch_size, height, width, channels)
    # Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
    # jointly over the image dimensions
    H = heigth
    W = width
    C = channels
    temp = 0.01
    features = tf.reshape(tf.transpose(features, [0, 3, 1, 2]), [-1, H * W])
    print(tf.math.argmax(features, axis = -1))
    # print(tf.math.reduce_max(features, axis = -1))
    softmax = tf.nn.softmax(features/temp, axis = -1)
    # print(softmax)
    print(tf.math.argmax(softmax, axis = -1))
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