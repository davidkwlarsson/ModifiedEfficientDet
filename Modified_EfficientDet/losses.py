# import tensorflow as tf
from tensorflow.keras import backend as K

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


def heatmap_loss(y_true, y_pred):
    l = ((y_pred-y_true)**2)
    l = K.mean(K.mean(K.mean(l, axis = -1), axis = -1), axis = -1)
    return l


def null_loss(y_true, y_pred):
    return K.zeros_like(y_true)




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


def get_bone_length_pen(xyz):
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

    bone_length_dict = create_bone_dict()

    penalty = []
  # print(xyz.shape)
    n = 0
    for keys in bone_length_dict.keys():
        i1 = keys[0]
        i2 = keys[1]
        j1 = xyz[i1]
        j2 = xyz[i2]
       # print('j1',j1)

        dist = K.sum(K.square(j1-j2), axis = -1)
        min_val = bone_interval_dict[keys][0]
        max_val = bone_interval_dict[keys][1]
        #print('dist',K.eval(dist))
        gr = K.greater(dist, max_val)
        le = K.less(dist, min_val)
        #print(n)
        n+=1
        #K.switch(le,lambda: tf.add(min_val, -dist)), pen.append(0))
        #K.switch(gr,lambda: pen.append(tf.add(dist, max_val)), pen.append(0))
       #TODO: här blir det fel :(
        penalty.append(K.switch(
            le, 1.0,0.0#K.sum((min_val, -dist)), K.zeros(shape=(1))
        ))
        penalty.append(K.switch(
            gr, 1.0,0.0#K.sum((dist, -max_val)),  K.zeros(shape=(1))
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

def mse_bone_length_loss(y_true, y_pred):
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

    pred_bone_length_dict = create_bone_dict()
    #print('y_pred ', y_pred) # (None, 63)
    #print('y_pred ', y_pred[0])  # (None, 63)

    # loop?
    bone_penalty = 0
    xyz = K.reshape(y_pred, (-1, 21,3))
    bone_penalty = K.sum(tf.map_fn(get_bone_length_pen, xyz))
    # for i in range(batch_size):
    #   #  print('i', i)
    #     xyz = K.reshape(y_pred[i], (21,3))
    #     bone_penalty += get_bone_length_pen(pred_bone_length_dict, xyz, bone_interval_dict)
    # mse = K.mean(K.square((y_true-y_pred)))
    mse = tf.keras.losses.MSE(y_true,y_pred)
    lambd = 1
    print(bone_penalty)
    return mse + lambd*bone_penalty




















































import math as m
import tensorflow as tf
def norm_p(i, b_F):
    n, norm = tf.linalg.normalize(tf.linalg.cross(b_F[i+1], b_F[i]))
    return n

def angle(v1, v2):
    # print("...  : ", tf.tensordot(v1,v2,1))
    # print(".... : ",  tf.norm(v1)*tf.norm(v2))
    alpha = tf.math.acos(tf.tensordot(v1,v2, 1)/ \
        (tf.norm(v1)*tf.norm(v2)))
    return alpha

def ortho_project(x,y, v):
    A = tf.stack([x,y],axis = -1)
    P = tf.linalg.matvec(A, tf.linalg.matvec(
            tf.linalg.inv(tf.linalg.matmul(A,A, transpose_a = True)),
            tf.linalg.matvec(A,x,transpose_a = True)))
    # orthogonal projection of v onto vectors x and y
    return P
def angle_loss():
    def construct_b(y_pred):
        # vector number 
        b = []
        b_F = []
        # print(y_pred)
        y_pred = tf.reshape(y_pred,(-1,3))
        b_F.append(y_pred[1,:] - y_pred[0,:])
        for i in range(0,4):
            b.append(y_pred[i+1,:] - y_pred[i,:])
        b.append(y_pred[5,:] - y_pred[0,:])
        b_F.append(y_pred[5,:] - y_pred[0,:])
        for i in range(4,8):
            b.append(y_pred[i+1,:] - y_pred[i,:])
        b.append(y_pred[9,:] - y_pred[0,:])
        b_F.append(y_pred[9,:] - y_pred[0,:])
        for i in range(9,12):
            b.append(y_pred[i+1,:] - y_pred[i,:])
        b.append(y_pred[13,:] - y_pred[0,:])
        b_F.append(y_pred[13,:] - y_pred[0,:])
        for i in range(13,16):
            b.append(y_pred[i+1,:] - y_pred[i,:])
        b.append(y_pred[17,:] - y_pred[0,:])
        b_F.append(y_pred[17,:] - y_pred[0,:])
        for i in range(17,20):
            b.append(y_pred[i+1,:] - y_pred[i,:])

        return b, b_F
    
    def constuct_axis(b_F):
        xaxis = []
        yaxis = []
        zaxis = []
        for i in range(len(b_F)):
            zaxis.append(tf.linalg.normalize(b_F[i])[0])
            if i in (0,1):
                xaxis.append(-1*norm_p(i,b_F))
            if i in (2,3):
                xaxis.append(-tf.linalg.normalize(norm_p(i,b_F)-norm_p(i-1,b_F))[0])
            if i == 4:
                xaxis.append(-1*norm_p(3,b_F))
            yaxis.append(tf.norm(tf.linalg.cross(zaxis[i], xaxis[i])))

        return xaxis,yaxis,zaxis
            
    
    def calc_angle(b, x,y,z):
        theta_a = []
        theta_f = []
        # print("b : ", len(b))
        # print("x : ", len(x))
        # print("y : ", len(y))
        # print("z : ", len(z))
        for i in range(len(b)):
            if i not in [0,1,5,9,13,17]:
                if i in [2,3,4]:
                    f_i = 0
                elif i in [6,7,8]:
                    f_i = 1
                elif i in [10,11,12]:
                    f_i = 2
                elif i in [14,15,16]:
                    f_i = 3
                elif i in [18,19,20]:
                    f_i = 4

                theta_a.append(angle(ortho_project(x[f_i], z[f_i], b[i]), z[f_i]))
                theta_f.append(angle(ortho_project(x[f_i], z[f_i], b[i]), b[i]))
        return theta_a, theta_f

    def D_H(theta_a, theta_f):
        max_f = m.pi
        max_a = m.pi/2
        # print("theta_a : ", theta_a)
        # print("theta_f : ", theta_f)
        dist_a = tf.math.abs(theta_a) - max_a
        dist_f = tf.math.abs(theta_f) - max_f
        zero = tf.constant(0, dtype = tf.float32)
        dist_a = tf.cond(tf.math.greater(dist_a,0), lambda : dist_a, lambda : tf.constant(0, dtype = tf.float32))
        dist_f = tf.cond(tf.math.greater(dist_f,0), lambda : dist_f, lambda : tf.constant(0, dtype = tf.float32))
        return dist_a + dist_f
    
    def loss(y_true, y_pred):
        b, b_F = construct_b(y_pred)
        x,y,z = constuct_axis(b_F)
        theta_a, theta_f = calc_angle(b,x,y,z)
        L_a = 0
        for i in range(15):
            L_a += D_H(theta_a[i], theta_f[i])

        mse_loss = tf.keras.losses.MSE(y_true, y_pred)
        # mse_loss = K.mean(K.square(y_true-y_pred))

        return mse_loss + 1000*L_a/15  
        
    return loss