import os
import sys
import pickle
import matplotlib
import math
import json

import numpy as np
from numpy import asarray
from numpy import savetxt
from utils.fh_utils import *
import skimage.io as io

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf



def render_onehot_heatmap(coord, output_shape, num_keypoints):
    batch_size = tf.shape(coord)[0]
    input_shape = output_shape

    x = tf.reshape(coord[:, :, 0] / input_shape[1] * output_shape[1], [-1])
    y = tf.reshape(coord[:, :, 1] / input_shape[0] * output_shape[0], [-1])
    x_floor = tf.floor(x)
    y_floor = tf.floor(y)

    x_floor = tf.clip_by_value(x_floor, 0, output_shape[1] - 1)  # fix out-of-bounds x
    y_floor = tf.clip_by_value(y_floor, 0, output_shape[0] - 1)  # fix out-of-bounds y

    indices_batch = tf.expand_dims(tf.to_float( \
        tf.reshape(
            tf.transpose( \
                tf.tile( \
                    tf.expand_dims(tf.range(batch_size), 0) \
                    , [num_keypoints, 1]) \
                , [1, 0]) \
            , [-1])), 1)
    indices_batch = tf.concat([indices_batch] * 4, axis=0)
    indices_joint = tf.to_float(tf.expand_dims(tf.tile(tf.range(num_keypoints), [batch_size]), 1))
    indices_joint = tf.concat([indices_joint] * 4, axis=0)

    indices_lt = tf.concat([tf.expand_dims(y_floor, 1), tf.expand_dims(x_floor, 1)], axis=1)
    indices_lb = tf.concat([tf.expand_dims(y_floor + 1, 1), tf.expand_dims(x_floor, 1)], axis=1)
    indices_rt = tf.concat([tf.expand_dims(y_floor, 1), tf.expand_dims(x_floor + 1, 1)], axis=1)
    indices_rb = tf.concat([tf.expand_dims(y_floor + 1, 1), tf.expand_dims(x_floor + 1, 1)], axis=1)
    indices = tf.concat([indices_lt, indices_lb, indices_rt, indices_rb], axis=0)
    indices = tf.cast(tf.concat([tf.to_float(indices_batch), tf.to_float(indices), tf.to_float(indices_joint)], axis=1),
                      tf.int32)

    prob_lt = (1 - (x - x_floor)) * (1 - (y - y_floor))
    prob_lb = (1 - (x - x_floor)) * (y - y_floor)
    prob_rt = (x - x_floor) * (1 - (y - y_floor))
    prob_rb = (x - x_floor) * (y - y_floor)
    probs = tf.concat([prob_lt, prob_lb, prob_rt, prob_rb], axis=0)
    #  probs = tf.concat([prob_lt,prob_lt,prob_lt,prob_lt], axis=0)

    heatmap = tf.scatter_nd(indices, probs, (batch_size, *output_shape, num_keypoints))
    normalizer = tf.reshape(tf.reduce_sum(heatmap, axis=[1, 2]), [batch_size, 1, 1, num_keypoints])
    normalizer = tf.where(tf.equal(normalizer, 0), tf.ones_like(normalizer), normalizer)
    heatmap = heatmap / normalizer

    return heatmap


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K): 
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def plot_heatmaps_with_coords(images, heatmaps, coords):
    # Here I display heatmaps and coordinates to check that the heatmaps are correctly generated
    #get data
    #dir_path = sys.argv[1]
    #images, heatmaps, coords = get_trainData(dir_path,multi_dim=True)
    hm = heatmaps[0][:, :, 0]
    for i in range(1,21):
        hm += heatmaps[0][:, :, i]
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(images[0])
    plt.scatter(coords[0][0::2], coords[0][1::2], marker='o', s=2)
    plt.subplot(1, 2, 2)
    plt.imshow(hm)
    plt.scatter(coords[0][0::2], coords[0][1::2], marker='o', color='r', s=2)
    plt.colorbar()
    plt.show()
    print(coords[0][0:2])


def plot_acc_loss(history):
    # Plot acc and loss vs epochs
    # Plot acc and loss vs epochs
    #acc = history.history['acc']
    #val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
   # plt.plot(epochs, acc, 'bo', label='Training acc')
   # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    #plt.title('Training and validation accuracy')
    #plt.legend()
    #plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    plt.savefig('acc_loss.png')


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version))
    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)


def heatmaps_to_coord(heatmaps):
    coords = [[] for x in range(len(heatmaps))]
    for j in range(len(heatmaps)):
        for i in range(21):
            ind = np.unravel_index(np.argmax(heatmaps[j][:, :, i], axis=None), heatmaps[j][:, :, i].shape)
            coords[j].append(ind[1])
            coords[j].append(ind[0])

    return np.array(coords)

def get_depthmaps(uv,xyz_list):
    depths = np.zeros((224,224,21))
    xyz = np.array(xyz_list)
    for j,coord in enumerate(uv):
        try:
            depths[int(coord[0]),int(coord[1]),j] = xyz[j, 2]
        except:
            print("\n")

    return depths


def create_gaussian_hm(uv, w, h):
    # TODO: Hantera kantfallen också
    hm_list = list()
    im = np.zeros((w, h))
    std = 100
    for coord in uv:
        u = coord[1]
        v = coord[0]
        radius = 30
        hm_small = np.zeros((radius * 2 + 1, radius * 2 + 1))
        xc, yc = (radius + 1, radius + 1)
        for x in range(radius * 2 + 1):
            for y in range(radius * 2 + 1):
                dist = math.sqrt((x - xc) ** 2 + (y - yc) ** 2)
                if dist < 30:
                    scale = 10  # otherwise predict only zeros
                    hm_small[x][y] = scale * math.exp(-dist ** 2 / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))
        # plt.imshow(hm_small)
        # plt.show()
        # print('here')
        try:
            xc_im = np.round(u)
            yc_im = np.round(v)
            im[int(xc_im - radius - 1):int(xc_im + radius), int(yc_im - 1 - radius):int(yc_im + radius)] = hm_small
        except:
            print('Gaussian hm failed')
            continue
        hm_list.append(im)
        im = np.zeros((w, h))
    # plt.imshow(im)
    # plt.show()
    return np.transpose(np.array(hm_list), (1, 2, 0))

def create_onehot(uv, w, h):
    # heats = list()
    uv = uv[:, ::-1]
    temp_im = np.zeros((w,h,21))
    # temp_im2 = np.zeros((w*2,h*2,21))
    # temp_im3 = np.zeros((w*4,h*4,21))
        # img = list()
    for j,coord in enumerate(uv):
            # temp_im = np.zeros((224,224))
        # try: 
        temp_im[int(coord[0]/8), int(coord[1]/8),j] = 1
            # temp_im2[int(coord[0]/2), int(coord[1]/2),j] = 1
            # temp_im3[int(coord[0]), int(coord[1]),j] = 1
        # except:
        #     print("\n Coordinates where out of range : " , coord[0], coord[1])
    return temp_im #, temp_im2, temp_im3

def get_depth(xyz_list):
    depth = np.zeros(21)
    xyz = np.array(xyz_list)
    for j in range(21):
        depth[j] = xyz[j, 2] #- xyz[0,2]
    return depth



def save_preds(dir_path, predictions):
    #Function that saves away the predictions as jpg images.
    save_path = dir_path + "predictions/test"
    makedirs(save_path)
    print("Saving the predictions to " + save_path)
    predictions *= 255
    for i, pred in enumerate(predictions):
        name = save_path + str(i)+".jpg"
        io.imsave(name, pred.astype(np.uint8))


def plot_predicted_heatmaps(preds, heatmaps, images):
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 5
    n = 0
    for i in range(1,rows+1,2):
        fig.add_subplot(rows, columns, i)
        # plt.imshow(preds[0][:, :, n])
        pred = np.sum(preds[0], axis = -1)
        plt.imshow(pred)
        plt.colorbar()
        fig.add_subplot(rows, columns, i+1)
        # plt.imshow(heatmaps[0][:,:,n])
        plt.imshow(images[0])
        plt.colorbar()
        n += 1
        break
    plt.savefig('heatmaps.png')
    plt.show()



def plot_predicted_coordinates(images, coord_preds, coord):
    
    try:
        fig = plt.figure(figsize=(8, 8))
        columns = 5
        rows = 2
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.imshow(images[i - 1])
            # need to project onto image..
            plt.scatter(coord[i - 1][0::2], coord[i - 1][1::2], marker='o', s=2)
            plt.scatter(coord_preds[i - 1][0::2], coord_preds[i - 1][1::2], marker='x', s=2)
            # plt.scatter(coord[i - 1], marker='o', s=2)
            # plt.scatter(coord_preds[i - 1], marker='x', s=2)
            
        plt.savefig('scatter.png')
        plt.show()
    except:
        print('Error in scatter plot')



def plot_predicted_hands(images, coord_preds):
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(images[i - 1])
        # need to project onto image..
        one_pred = [coord_preds[i-1][0::2], coord_preds[i-1][1::2]]
        plot_hand(ax, np.transpose(np.array(one_pred)))

    plt.show()
    plt.savefig('hands.png')



def plot_predicted_hands_uv(images, coord_preds):
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(images[i - 1])
        # need to project onto image..
        one_pred = [coord_preds[i-1][0::2], coord_preds[i-1][1::2]]
        # one_pred = np.transpose(coord_preds[i-1])
        plot_hand(ax, np.transpose(np.array(one_pred)), order = 'uv')
        
    plt.savefig('hands_uv.png')
    plt.show()


def add_depth_to_coords(coords, depth, K_list):
    xyz = []
    ## Transform the x,y coordinates back to 3D using the intrinsic camera parameters, K
    
    for i in range(len(coords)):
        K = np.array(K_list[i])
        # depth[i] += depth[0]
        x_coords = coords[i][0::2]*depth[i]
        y_coords = coords[i][1::2]*depth[i]
        uv_z = np.array([x_coords, y_coords, depth[i]])
        # print(uv_z.shape, K.shape)
        xyz.append(np.array(np.linalg.solve(K,uv_z)).T)

    return xyz




### FOLLOWING CODE IS TAKEN FROM https://github.com/3d-hand-shape/hand-graph-cnn/blob/master/hand_shape_pose/util/vis.py

color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little


def draw_3d_skeleton(pose_cam_xyz, image_size):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(111, projection='3d')
    marker_sz = 15
    line_wd = 2

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # pickle.dump(fig, open('3DHands.fig.pickle', 'wb'))
    plt.savefig('hands_3D.png')

    # ret = fig2data(fig)  # H x W x 4
    # plt.close(fig)
    # return ret

    # figx = pickle.load(open('3DHands.fig.pickle', 'rb'))
    # print(" Opening Pickle file --- ")
    # figx.show() # Show the figure, edit it, etc.!
    plt.show()



def save_coords(pose_cam_coord, image):
    pose_cam_coord = np.array(pose_cam_coord)
    l,n,m = pose_cam_coord.shape
    pose_cam_coord = pose_cam_coord.reshape((l*n, m))
    if pose_cam_coord.shape[-1] == 3:
        savetxt('pose_cam_xyz.csv',pose_cam_coord, delimiter=',')
        pickle.dump(image, open('hand_for_3d.fig.pickle', 'wb'))
    elif pose_cam_coord.shape[-1] == 2:
        savetxt('pose_cam_xy.csv',pose_cam_coord, delimiter=',')
        pickle.dump(image, open('hand_for_2d.fig.pickle', 'wb'))




def save_model(model):
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')
    print('Saved model to Disk! ')

from tensorflow.keras.models import model_from_json
from keypointconnector import ConnectKeypointLayer
from layers import BatchNormalization, wBiFPNAdd
from efficientnet import get_swish, get_dropout
from tensorflow.keras import layers


def load_model(dir = None,print_model = False):
    if dir == None:
        json_file = open('model.json', 'r')
        h5model = 'model.h5'
    else:
        path = os.path.join(dir, 'model.json')
        json_file = open(path, 'r')
        h5model = os.path.join(dir, 'model.h5')

    loaded_model_json = json_file.read()
    json_file.close()
    print(loaded_model_json)
    layer_dict = {'ConnectKeypointLayer' : ConnectKeypointLayer,
                'BatchNormalization' : BatchNormalization,
                'wBiFPNAdd' : wBiFPNAdd,
                'swish': get_swish(),
                'FixedDropout' : get_dropout()}
    loaded_model = model_from_json(loaded_model_json, layer_dict)
    loaded_model.load_weights(h5model)
    print('Loaded model ')
    if print_model == True:
        print(loaded_model.summary())

    return loaded_model












# def get_trainData(dir_path, num_samples, multi_dim = True):
#     print("Collecting data ... \n")
#     imgs = []
#     heats = []
#     heats2 = []
#     heats3 = []
#     coords = []
#     xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
#     K_list = json_load(os.path.join(dir_path, 'training_K.json'))
#     for i in range(num_samples):
#         # load images
#         img = read_img(i, dir_path, 'training')
#         imgs.append(img)
#         # project 3d coords and create heatmaps
#         uv = projectPoints(xyz_list[i], K_list[i])
#         # heats.append(create_gaussian_hm(uv,56,56))
#         onehots = create_onehot(uv, 56,56)
#         heats.append(onehots[0])
#         heats2.append(onehots[1])
#         heats3.append(onehots[2])
#         coords.append([])
#         for j,coord in enumerate(uv):
#             # save coordinates
#             coords[i].append(coord[0])
#             coords[i].append(coord[1])

#     return np.array(imgs), np.array(heats), np.array(heats2), np.array(heats3), np.array(coords)


# def get_evalImages(dir_path, num_samples):
#     print("Collecting data evaluation data ... \n")
#     imgs = []
#     for i in range(num_samples):
#         # load images
#         img = read_img(i, dir_path, 'evaluation')
#         imgs.append(img)

#     return np.array(imgs)
