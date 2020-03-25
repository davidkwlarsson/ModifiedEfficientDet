# import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from lift_pose import projectPoints
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


def weighted_bce(y_true, y_pred):
    weights = (y_true * 59.) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce


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

