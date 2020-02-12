# import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K

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