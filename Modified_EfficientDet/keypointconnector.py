import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.compat.v1.keras import backend as K

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



    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = list()
        for i in range(21):
            self.W.append(self.add_weight(name='lambda',
                                     shape=[32,
                                      32],
                                      initializer='uniform',
                                     trainable=True))

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
            l = K.dot(g,self.W[i])
            fn.append(K.sum(K.concatenate((f[i], l), axis=-1), axis=-1, keepdims=True))

        return K.concatenate(tuple(fn), axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)