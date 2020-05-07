import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from utils.fh_utils import projectPoints
import numpy as np
def binary_focal_loss(gamma = 2., alpha = 0.25):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1.-pt_1, gamma)*K.log(pt_1)) \
            -K.sum((1-alpha) * K.pow(pt_0, gamma)*K.log(1.-pt_0))
        
    return focal_loss_fixed


def categorical_focal_loss(gamma = 2., alpha = 0.25):

    def cat_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis = (1,2), keepdims =True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.-epsilon)

        cross_entropy = -y_true*K.log(y_pred)
        loss = alpha * K.pow(1-y_pred, gamma) * cross_entropy


        return K.sum(loss)
    return cat_focal_loss_fixed

# class weighted_bce(tf.keras.losses.Loss):
#     def __init__(self, name='weighted_bce'):
#         super().__init__(name=name)
#        # self.pos_weight = pos_weight
#
#     def call(self, y_true, y_pred):
#         weights = (y_true * 59.) + 1.
#         bce = K.binary_crossentropy(y_true, y_pred)
#         weighted_bce = K.mean(bce * weights)
#         return weighted_bce

def weighted_bce(y_true, y_pred):
    weights = (y_true * 59.) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce


def weighted_mse(y_true, y_pred):
    xyz = K.reshape(y_pred,(21,3))
    z = xyz[:,-1]
    xy = K.stack(xyz[:,:-1]/K.transpose([z,z]))
    print(xy)
    #xyz = K.reshape(xy, (42,))
    xyz = K.flatten(xy)
    print('xyz',xyz)
    print('xyz',y_true)
    error = K.mean(K.abs(K.square(xyz - y_true)))
    print(error)
    return error

def create_bone_dict():
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
            connection_dict[i].append(i - 1)
        if i not in [0, 4, 8, 12, 16, 20]:
            connection_dict[i].append(i + 1)

    # print(connection_dict)
    bone_length_dict = dict()
    pred_bone_length_dict = dict()
    for i in range(21):
        for j in range(i, 21):
            if j in connection_dict[i]:
                bone_length_dict[(i, j)] = []
                pred_bone_length_dict[(i, j)] = []
    return pred_bone_length_dict


def get_bone_length_pen(bone_length_dict, xyz, bone_interval_dict):
    penalty = []
  # print(xyz.shape)
    n = 0
    for keys in bone_length_dict.keys():
        i1 = keys[0]
        i2 = keys[1]
        j1 = xyz[i1]
        j2 = xyz[i2]

        #print('j1',j1)
        #dist = K.abs(j1-j2)
        dist = K.sqrt(K.sum(K.square(j1-j2)))
        min_val = bone_interval_dict[keys][0]
        max_val = bone_interval_dict[keys][1]
        #print('dist',K.eval(dist))
        gr = K.greater(
            dist, max_val
        )
        le = K.less(
            dist, min_val
        )
        #print(n)
        n+=1
        #K.switch(le,lambda: tf.add(min_val, -dist)), pen.append(0))
        #K.switch(gr,lambda: pen.append(tf.add(dist, max_val)), pen.append(0))
        #TODO: här blir det fel :(
        penalty.append(K.switch(
            le, K.sum((min_val, -dist)), 0.0#K.sum((min_val, -dist)), K.zeros(shape=(1))
        ))
        penalty.append(K.switch(
            gr, K.sum((dist, -max_val)), 0.0#K.sum((dist, -max_val)),  K.zeros(shape=(1))
        ))
       # print(le)
 # more than max

      #  if K.get_value(le): # less than min
      #      pen.append(K.sum((min_val,-dist)))
      #  elif K.get_value(gr):  # more than max
      #      pen.append(K.sum((dist,-max_val)))
    penalty = K.stack(penalty)
   # print('pen',pen)
   # print('K.mean(pen)',K.mean(pen))
    return K.mean(penalty)

def mse_bone_length_loss(batch_size=16):
    def mse_bone_length(y_true, y_pred):
        bone_interval_dict = {(0, 1): (1.0908192232950888, 2.2515338614776392), (0, 5): (2.669800710572548, 4.794494704465676),
         (0, 9): (2.8136933580398042, 4.780820266037302), (0, 13): (2.4398277270577546, 4.485480161197701),
         (0, 17): (2.2869092176287666, 4.366607255536491), (1, 2): (0.7499442490686489, 1.774474642908402),
         (2, 3): (0.6654086896008954, 1.6097315654018196), (3, 4): (0.9601543205501539, 1.9852891832644235),
         (5, 6): (0.6894040216476865, 1.7067186969939592), (6, 7): (0.41366510750293395, 1.177975645406827),
         (7, 8): (0.7775762513501541, 1.3375914440827075), (9, 10): (0.9999999999999978, 1.0000000000000027),
         (10, 11): (0.5872641243720444, 0.909916636318616), (11, 12): (0.7976008278141293, 1.340945662500335),
         (13, 14): (0.5789761966270687, 1.3134559130392942), (14, 15): (0.5257988175827191, 1.1861404804200057),
         (15, 16): (0.7407154674361921, 1.332961289773149), (17, 18): (0.34066985651664144, 1.1479969524877807),
         (18, 19): (0.341630422983034, 0.9554632024043269), (19, 20): (0.5741511747264161, 1.0941401493395624)}
      #  bone_interval_dict = {(0, 1): (0, 5),
      #                        (0, 5): (0, 5),
      #                        (0, 9): (0, 5),
      #                        (0, 13): (0, 5),
      #                        (0, 17): (0, 5),
      #                        (1, 2): (0, 5),
      #                        (2, 3): (0, 5),
      #                        (3, 4): (0, 5),
      #                        (5, 6): (0, 5),
       #                       (6, 7): (0, 5),
       #                       (7, 8): (0, 5),
       ##                       (9, 10): (0, 5),
       ##                       (10, 11): (0, 5),
       ##                       (11, 12): (0, 5),
       ##                       (13, 14): (0, 5),
       ##                       (14, 15): (0, 5),
       ##                       (15, 16): (0, 5),
       ##                       (17, 18): (0, 5),
        #                      (18, 19): (0, 5),
        #                      (19, 20): (0, 5)}
        pred_bone_length_dict = create_bone_dict()
        #print('y_pred ', y_pred) # (None, 63)
        #print('y_pred ', y_pred[0])  # (None, 63)

        # loop?
        bone_penalty = 0
        for i in range(batch_size):
          #  print('i', i)
            xyz = K.reshape(y_pred[i], (21,3))
            bone_penalty += get_bone_length_pen(pred_bone_length_dict, xyz, bone_interval_dict)
        mse = K.mean(K.abs((y_true,y_pred)))
        lambd = 1
        print(bone_penalty)
        return mse + lambd*bone_penalty
    return mse_bone_length

def lift_loss(y_true, y_pred):
    lambd = 1000.
    z = K.mean(K.abs(y_pred[0]-y_true[0]))
    P = lambd*K.mean(K.abs(y_pred[1:]-y_true[1:]))
    return z + P


# only began to think about this..
def projected_error(K):
    def projected_error(y_true, y_pred):
        y_t = projectPoints(y_true, K)
        y_p = projectPoints(y_pred, K)
        e = K.mse(y_t, y_p)
        return e
    return projected_error

