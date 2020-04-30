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



def plot_acc_loss(history, im_size = 224):
    dirName = 'saved_xyz_data_' + str(im_size) + '.csv'
    # if xyz_only:
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    # plt.savefig(os.path.join(dirName,'acc_xyz_loss.png'))
    print(history.history.keys())
    loss = history.history['loss']
    xyz_loss = history.history['xyz_loss']
    val_xyz_loss = history.history['val_xyz_loss']
    val_loss = history.history['val_loss']
    lr = history.history['lr']
    epochs = range(1, len(loss) + 1)
    np.savetxt(os.path.join(dirName,'loss.csv'), loss, delimiter=',')
    np.savetxt(os.path.join(dirName,'xyz_loss.csv'), xyz_loss, delimiter=',')
    np.savetxt(os.path.join(dirName,'val_loss.csv'), val_loss, delimiter=',')
    np.savetxt(os.path.join(dirName,'val_xyz_loss.csv'), val_xyz_loss, delimiter=',')
    np.savetxt(os.path.join(dirName,'epochs.csv'), epochs, delimiter=',')
    np.savetxt(os.path.join(dirName,'lr.csv'), lr, delimiter=',')
    # else:
    #     dirName = 'saved_total_data'
    #     try:
    #         # Create target Directory
    #         os.mkdir(dirName)
    #         print("Directory " , dirName ,  " Created ") 
    #     except FileExistsError:
    #         print("Directory " , dirName ,  " already exists")
    #     # plt.savefig(os.path.join(dirName,'acc_loss.png'))
    #     print(history.history.keys())
    #     loss = history.history['loss']
    #     xyz_loss = history.history['xyz_loss']
    #     val_xyz_loss = history.history['val_xyz_loss']
    #     val_loss = history.history['val_loss']
    #     lr = history.history['lr']
    #     epochs = range(1, len(loss) + 1)
    #     np.savetxt(os.path.join(dirName,'loss.csv'), loss, delimiter=',')
    #     np.savetxt(os.path.join(dirName,'xyz_loss.csv'), xyz_loss, delimiter=',')
    #     np.savetxt(os.path.join(dirName,'val_loss.csv'), val_loss, delimiter=',')
    #     np.savetxt(os.path.join(dirName,'val_xyz_loss.csv'), val_xyz_loss, delimiter=',')
    #     np.savetxt(os.path.join(dirName,'epochs.csv'), epochs, delimiter=',')
    #     np.savetxt(os.path.join(dirName,'lr.csv'), lr, delimiter=',')



def plot_acc_loss_2D(history, im_size = 224):
    dirName = 'saved_2D_uv_data_' + str(im_size) + '.csv'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    # plt.savefig(os.path.join(dirName,'acc_uv_loss.png'))
    print(history.history.keys())
    loss = history.history['loss']
    uv_loss = history.history['uv_coords_loss']
    val_uv_loss = history.history['val_uv_coords_loss']
    val_loss = history.history['val_loss']
    lr = history.history['lr']
    epochs = range(1, len(loss) + 1)
    np.savetxt(os.path.join(dirName,'loss.csv'), loss, delimiter=',')
    np.savetxt(os.path.join(dirName,'uv_loss.csv'), uv_loss, delimiter=',')
    np.savetxt(os.path.join(dirName,'val_loss.csv'), val_loss, delimiter=',')
    np.savetxt(os.path.join(dirName,'val_uv_loss.csv'), val_uv_loss, delimiter=',')
    np.savetxt(os.path.join(dirName,'epochs.csv'), epochs, delimiter=',')
    np.savetxt(os.path.join(dirName,'lr.csv'), lr, delimiter=',')


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version))
    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)


def get_evalImages(dir_path, num_samples, dataset='training'):
    print("Collecting image data ...\n")
    imgs = []
    j = 0
    indicies=[]
    total_samples = 130240
    train_samples =int(total_samples / 20 * 17)
    validation_samples =int(total_samples / 10)
    test_samples =int(total_samples / 20)
    val_part =int(validation_samples / 4)
    test_part =int(test_samples / 4)
    train_part =int(train_samples / 4)
    total_part =int(total_samples / 4)
    if dataset == 'training':
        indicies = [i for i in range(train_part)] + \
                   [i for i in range(total_part, total_part+train_part)] + \
                   [i for i in range(total_part*2, total_part*2+train_part)] + \
                   [i for i in range( total_part*3, total_part*3+train_part)]
    if dataset == 'validation':
        indicies = [i for i in range(train_part, train_part+val_part)] + \
                   [i for i in range(total_part+train_part, total_part+train_part+val_part)] + \
                   [i for i in range(total_part*2+train_part, total_part*2+train_part+val_part)] + \
                   [i for i in range(total_part*3+train_part, total_part*3+train_part+val_part)]
    if dataset == 'test':
        indicies = [i for i in range(train_part+val_part, train_part+val_part+test_part)] + \
                   [i for i in range(total_part+train_part+val_part, total_part+train_part+val_part + test_part)] + \
                   [i for i in range(total_part*2+train_part+val_part, total_part*2+train_part+val_part + test_part)] + \
                   [i for i in range(total_part*3+train_part+val_part, total_part*3+train_part+val_part + test_part)]
    print('loading ', len(indicies[:num_samples]))
    indicies = indicies[:num_samples]
    for i in indicies:
        # load images
        img = read_img(i, dir_path, dataset)
        imgs.append(img)
    return np.array(imgs)


def get_depth(xyz_list):
    depth = np.zeros(21)
    xyz = np.array(xyz_list)
    for j in range(21):
        depth[j] = xyz[j, 2] #- xyz[0,2]
    return depth



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

def add_relative(xyz_pred, xyz_list, s_list):
    xyz_pred = np.reshape(xyz_pred, (-1,21,3))
    print(len(xyz_pred))
    xyz_new = []
    for i in range(len(xyz_pred)):

        xy = np.array(xyz_pred[i])[:, 0:-1]*s_list[i]
        z_pred = np.array(xyz_pred[i])[:,2]*s_list[i]
        z_tar = np.array(xyz_list[i])[:,2]
        z_pred = z_pred + z_tar[0]
        xyz = np.concatenate((xy, np.expand_dims(z_pred, axis=-1)),axis = -1)
        xyz = np.ndarray.flatten(xyz)
        xyz_new.append(xyz)
    return xyz_new

def add_depth_to_coords(coords, depth, K_list, s_list):
    xyz = []
    ## Transform the x,y coordinates back to 3D using the intrinsic camera parameters, K
    
    for i in range(len(coords)):
        K = np.array(K_list[i])
        depth[i] = depth[i]*s_list[i]
        x_coords = coords[i][0::2]*depth[i]
        y_coords = coords[i][1::2]*depth[i]
        uv_z = np.array([x_coords, y_coords, depth[i]])
        # print(uv_z.shape, K.shape)
        xyz.append(np.array(np.linalg.solve(K,uv_z)).T)

    return xyz



def save_coords(pose_cam_coord, image):
    pose_cam_coord = np.array(pose_cam_coord)
    # l,n,m = pose_cam_coord.shape
    # pose_cam_coord = pose_cam_coord.reshape((l*n, m))
    if pose_cam_coord.shape[-1] == 3*21:
        savetxt('pose_cam_xyz.csv',pose_cam_coord, delimiter=',')
        pickle.dump(image, open('hand_for_3d.fig.pickle', 'wb'))
    elif pose_cam_coord.shape[-1] == 2*21:
        savetxt('pose_cam_xy.csv',pose_cam_coord, delimiter=',')
        pickle.dump(image, open('hand_for_2d.fig.pickle', 'wb'))
    else:
        print("No coordinates saved!")
