import sys

import tensorflow as tf
from numpy import int64
from tensorflow.keras import layers, Input, Model, losses
from data_generators import *
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import matplotlib


def get_uncompiled_model():
  inputs = Input(shape=(224,224,3), name='digits')
  x = layers.Conv2D(21, kernel_size=1, activation='relu', name='dense_1')(inputs)
  model = Model(inputs=inputs, outputs=x)
  return model

def get_compiled_model():
  model = get_uncompiled_model()
  model.compile(optimizer=Adam(learning_rate=1e-3),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
  return model



def create_hm_tensors(uv_all, gaussian):
    hm_tensor = []
    i = 0
    hm = []
    for uv in uv_all:
        hm_tensor = []
        #hm_tensor.append([])
        for uv_k in uv:
            u = uv_k[0]
            v = uv_k[1]
            ur = tf.range(int(u) - radius - 1, int(u) + radius)
            vr = tf.range(int(v) - radius - 1, int(v) + radius)
            UR, VR = tf.meshgrid(ur, vr)
            indices = tf.cast(tf.stack([UR, VR], axis=-1), dtype=tf.int64)
            indices = tf.reshape(indices, [tf.size(UR), 2])
            gaussian = tf.reshape(gaussian, [tf.size(indices) // 2])
           # print(indices)
           # print(gaussian)
            with tf.Session() as sess:
                imgs = sess.run(indices)
           # print(imgs)
           # print(u, v)

            vec = tf.SparseTensor(indices=indices, values=gaussian, dense_shape=[56, 56])
            hm_tensor.append(vec)
            hm.append(tf.stack(hm_tensor))
        i += 1
        print(i)
        if i > 5:
            break

    return tf.stack(hm)

if __name__ == '__main__':
    #
    matplotlib.use('TkAgg')

    try:
        dir_path = sys.argv[1]
    except:
        dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"  #
    uv = get_uv_data(dir_path)
    std = 2
    radius = 2
    gaussian = create_gaussian_blob(std, radius)
    #plt.imshow(gaussian)
    #plt.show()

    hm_tensor = create_hm_tensors(uv, gaussian)

   # with tf.Session() as sess:
      #imgs = sess.run(hm_tensor)
    print(hm_tensor)
    #print(u, v)

    dataset = tf.data.Dataset.from_tensor_slices((hm_tensor, hm_tensor))
    print(dataset)
