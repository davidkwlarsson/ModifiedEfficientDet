
from tensorflow.keras import layers, activations, models
import numpy as np
from data_generators import *
from help_functions import *
from plot_functions import *
from losses import *
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.optimizers import Adam, SGD
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))


def lift_model(nbr_outputs):
    #https://arxiv.org/pdf/1910.12029v2.pdf
    input_shape = (42)
    uv_input = layers.Input(input_shape)
    r0 = layers.Dense(4096, activation = 'linear')(uv_input)
    r1 = r0
    for i in range(2):
        ## Residual block ##
        r1 = layers.BatchNormalization()(r1)
        r1 = layers.Dropout(0.5)(r1)
        r1 = activations.relu(r1)
        r1 = layers.Dense(4096, activation='linear')(r1)

    added = layers.Add()([r0, r1])

    depth = layers.Dense(nbr_outputs, activation='linear')(added)


    model = models.Model(inputs=[uv_input], outputs=[depth])

    return model


def plot_normalized_coords(uv, uv_n, images):
    coords = uv
    plt.subplot(1, 3, 1)
    plt.scatter(coords[:,0], coords[:,1], marker='o', s=2)
    plt.axis([0, 224, 0, 224])
    plt.subplot(1, 3, 3)
    plt.imshow(images)
    plt.scatter(coords[:,0], coords[:,1], marker='o', s=2)
    plt.subplot(1, 3, 2)
    coords = uv_n
    plt.scatter(coords[:, 0], coords[:, 1], marker='o', s=2)
    plt.axis([-1, 1, -1, 1])

    plt.show()


def relative_depth(z):
    z_r = []
    for p in z:
        p = p-p[0]
        z_r.append(p)
    return z_r


def normalize(uv_list, input_shape):
    S = np.array([[2/input_shape[0], 0 , 0],[0, 2/input_shape[1], 0],[0, 0, 1] ])
    T = np.array([[1, 0, -input_shape[0]/2],[0, 1, -input_shape[1]/2],[0,0,1]])
    N = np.matmul(S, T)
    uv_new_list = []
    for i in range(len(uv_list)):
        uv_new = []
        for uv in uv_list[i]:
            uv_new.append(np.matmul(N, np.array([uv[0], uv[1], 1])))
        uv_new_list.append(np.array(uv_new)[:, :2])
    return np.array(uv_new_list)


def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
   # print(uv)
    return uv[:, :2] / uv[:, -1:]


def reshape_data(uv_list):
    uv_new_list = []
    for i in range(len(uv_list)):
        uv_new = []
        for uv in uv_list[i]:
            uv_new.append(uv[0])
            uv_new.append(uv[1])
        uv_new_list.append(uv_new)
    return np.array(uv_new_list)


def reshape_target(xyz_list):
    xyz_new_list = []
    for i in range(len(xyz_list)):
        xyz_new = []
        for xyz in xyz_list[i]:
            xyz_new.append(xyz[0])
            xyz_new.append(xyz[1])
            xyz_new.append(xyz[2])
        xyz_new_list.append(xyz_new)
    return np.array(xyz_new_list)


def shape_target(xyz_list):
    xyz_new_list = []
    for i in range(len(xyz_list)):
        xyz_new = []
        n = 0
        for j in range(0,21*3,3):
            xyz_new.append([])
            xyz_new[n].append(xyz_list[i][j])
            xyz_new[n].append(xyz_list[i][j+1])
            xyz_new[n].append(xyz_list[i][j+2])
            n += 1
        xyz_new_list.append(xyz_new)
    return np.array(xyz_new_list)


def tmp_calc(uv, K):
    input_shape = (224,224)
    S = np.array([[2/input_shape[0], 0 , 0],[0, 2/input_shape[1], 0],[0, 0, 1] ])
    T = np.array([[1, 0, -input_shape[0]/2],[0, 1, -input_shape[1]/2],[0,0,1]])
    N = np.matmul(S, T)
    P = np.matmul(N, K)
    all = []
    for i in range(21):
        xyz = np.array([uv[i][0], uv[i][1], 1])
        right = np.matmul(np.linalg.inv(P), xyz)
        all.append(right)
    return all


def calc_scale(uv_all, K, depth):
    input_shape = (224,224)
    S = np.array([[2/input_shape[0], 0 , 0],[0, 2/input_shape[1], 0],[0, 0, 1] ])
    T = np.array([[1, 0, -input_shape[0]/2],[0, 1, -input_shape[1]/2],[0,0,1]])
    N = np.matmul(S, T)
    all = []
    for (i,uv) in enumerate(uv_all):
        P = np.array(np.matmul(N, np.array(K[i])))
        all.append([])
        for j in range(21):
            xyz = np.array([uv[j][0], uv[j][1], 1])
            xyz_3d = np.dot(depth[i][j], np.matmul(np.linalg.inv(P), xyz))
            all[i].append(xyz_3d)
    return np.array(all)


def extract_z(xyz_list):
    z = []
    z_root = []

    for idx in range(len(xyz_list)):
        z.append(np.array(xyz_list[idx])[:, 2])
        #z_root.append([])
        #for i in range(21):
        z_root.append(np.array(xyz_list[idx])[0, 2])
    return (z, z_root)


def extract_focal(K_list):
    f = []
    for i in range(len(K_list)):
        fx = np.array(K_list[i])[0][0]/224
        fy = np.array(K_list[i])[1][1]/224
     #   print(K_list[i])
     #   print(fx)
      #  print(fy)
        f.append(np.mean((fx, fy)))
    return f


def get_canonical(z_root, f):
    z_canonical = []
    for i in range(len(z_root)):
        z_canonical.append(z_root[i]/f[i])
    return z_canonical


def concat_uv_z(uv_n_list, z_rel, z_root_c):
    vec = []
    for i in range(len(uv_n_list)):
        vec.append([])
        vec[i].append(z_root_c[i])
        for j in range(1,21):
            vec[i].append(uv_n_list[i][j][0])
            vec[i].append(uv_n_list[i][j][1])
            vec[i].append(z_rel[i][j])
    return vec


def calc_canonical(uv_n_list_val, preds, c, f):
    xyz_all = []
    for n in range(len(preds)):
        xyz = []
        i = 0
        z_r = preds[i][0]
        rx = uv_n_list_val[i][0][0]
        ry = uv_n_list_val[i][0][1]
        Rx = (rx - c) * z_r
        Ry = (ry - c) * z_r
        Rz = z_r * f[i]
        xyz.append([Rx, Ry, Rz])
        Abs_z_root = Rz
        for i in range(1,len(preds[0]),3):
          #  print(i)
            Rx = preds[n][i]
            Ry = preds[n][i+1]
            Rz = preds[n][i+2]+Abs_z_root

            xyz.append([Rx, Ry, Rz])
        xyz_all.append(xyz)
    return xyz_all


def main():
    try:
        dir_path = sys.argv[1]
    except:
        dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"  #
    print('in correct file')
   # matplotlib.use('TkAgg')
   # data_set = 'small_dataset'


    data_set = 'training'
    width = 224
    height = 224
    input_shape = (width, height)
    xyz_list, K_list, num_samples = get_raw_data(dir_path, data_set)
    xyz_list_val, K_list_val, num_samples_val = get_raw_data(dir_path, 'validation')
    batch_size = 16
    nbr_epochs = 60
    # This data should come from network later..
    uv_list = get_uv_data(xyz_list, K_list, num_samples)
    uv_list_val = get_uv_data(xyz_list_val, K_list_val, num_samples_val)
    # Normalize so center in (0,0) and corners in (+-1,+-1)
    uv_n_list = normalize(uv_list, input_shape)
    uv_n_list_val = normalize(uv_list_val, input_shape)
    # Extract focal length from K matrix
    f = extract_focal(K_list)
    f_val = extract_focal(K_list_val)
    # Extract z coordinate and root
    z, z_root = extract_z(xyz_list)
    z_val, z_val_root = extract_z(xyz_list_val)
    # Divide root with focal length so indep. of camera
    z_root_c = get_canonical(z_root, f)
    z_val_root_c = get_canonical(z_val_root, f_val)
    # Calculate relative pose coordinates
    z_rel = relative_depth(z)
    z_rel_val = relative_depth(z_val)


    # Creating 3J-2 outputs

    target = np.array(concat_uv_z(uv_n_list, z_rel, z_root_c))
    target_val = np.array(concat_uv_z(uv_n_list_val, z_rel_val, z_val_root_c))
    #print(target)
    #target = np.array(z_rel)
    #target_val = np.array(z_rel_val)
    data = reshape_data(uv_n_list)
    data_val = reshape_data(uv_n_list_val)
    print('target', np.shape(target))
    print('target_val', np.shape(target_val))
    print('data', np.shape(data))
    print('data_val', np.shape(data_val))
    model = lift_model(61)
    model.compile(optimizer=Adam(lr=1e-3), loss=lift_loss)#loss='mse')
    model.summary()
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(x=data, y=target, validation_data=(data_val, target_val),
                        validation_steps=num_samples_val // batch_size,
                        steps_per_epoch=num_samples // batch_size,
                        epochs=nbr_epochs, verbose=1, callbacks=[callback])
    plot_acc_loss(history)
    preds = model.predict(data_val)
    print(np.shape(preds))
    im_center = 0
    xyz_preds = calc_canonical(uv_n_list_val, preds, im_center, f_val)
    print('Preds reshaped',np.shape(xyz_preds))
    xyz_tar = xyz_list_val


    images = get_evalImages(dir_path, 10, dataset='validation')

    for i in range(10):
        save_coords(xyz_preds[i], images[i], 'pred_' + str(i))
        save_coords(xyz_tar[i], images[i], 'target_' + str(i))

if __name__ == '__main__':
    tf.keras.backend.clear_session()
   # use_multiprocessing = True
    main()