import os
import sys
import pickle
import matplotlib
import math
from skimage.transform import resize
import numpy as np
from utils.fh_utils import *
import skimage.io as io
from generator import projectPoints, json_load, _assert_exist
from data_generators import create_gaussian_blob
import cv2
from csv import writer
from csv import reader
from numpy import genfromtxt

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.sparse import lil_matrix


from skimage import img_as_ubyte
from numpy import asarray
from numpy import savetxt

def evaluate_result(dir_path, preds):
    predictions = []
    n = 0

    # Sum all heatmaps for the joints of one hand and save as figure
    for sample in preds:
        predictions.append(sample)
        sum = 0
        for k in range(21):
            #  print(np.shape(sample))
            sum = sum + sample[:, :, k]
        plt.figure()
        plt.imshow(sum)
        plt.colorbar()
        plt.savefig('tf_sum_fig_preds_hm' + str(n) + '.png')
        n = n + 1

    # From the predictions extract coordinates
    coord_preds = heatmaps_to_coord(predictions)
    # Get the images that corresponds to test data
    # Get the true coordinates
    images, uv, hm, c = get_data(dir_path, 10)
    coord = uv
    #images = get_evalImages(dir_path, 10)
    # Plot the predicted result as skeleton
    plot_predicted_hands_uv(images, coord_preds, 'hands_as_uv_pred.png')
    # Plot the images together with true and predicted heatmap
    plot_hm_with_images(predictions, predictions, images, 4)
    # bplot of hand
    plot_predicted_coordinates(images, coord_preds, coord)



def plot_heatmaps_with_coords(images, heatmaps, coords):
    """Here the heatmaps are displayed together with the coordinates to
    check that the heatmaps are correctly generated"""
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
  #print(coords[0][0:2])
    plt.show()


def plot_acc_loss(history):
    """Plot acc and loss vs epochs"""
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
    """ Take max of heatmap and return its coordinates"""
    coords = [[] for x in range(len(heatmaps))]
    for j in range(len(heatmaps)):
        for i in range(21):
            ind = np.unravel_index(np.argmax(heatmaps[j][:, :, i], axis=None), heatmaps[j][:, :, i].shape)
            coords[j].append(ind[1])
            coords[j].append(ind[0])

    return np.array(coords)

def create_gaussian_hm(uv, w, h, radius, hm_small):
    hm_list = list()
    hm_list2 = list()
    hm_list3 = list()
    py = 50
    px = 50
    im = np.zeros((w+2*px, h+2*py))
    for coord in uv:
        u = coord[0]+px
        v = coord[1]+py
        try:
            xc_im = np.round(u)
            yc_im = np.round(v)
            im[int(xc_im - radius - 1):int(xc_im + radius), int(yc_im - 1 - radius):int(yc_im + radius)] = hm_small
        except:
            print('Gaussian hm failed\n')
            print(u, v)
            print(int(xc_im - radius - 1), int(xc_im + radius), int(yc_im - 1 - radius), int(yc_im + radius))
            print(coord[0], coord[1])
            print(im[int(xc_im - radius - 1):int(xc_im + radius), int(yc_im - 1 - radius):int(yc_im + radius)].shape)
            print(hm_small.shape)
            print(im[px:-px, py:-py].shape)
            print(im.shape)
            continue

        hm_list.append(im[px:-px, py:-py])
        hm_list2.append(resize(im, (w / 2, h / 2)))
        hm_list3.append(resize(im, (w / 4, h / 4)))
        im = np.zeros((w + 2 * px, h + 2 * py))

    # plt.imshow(im)
    # plt.show()
   # return np.array(hm_list)
    return np.transpose(np.array(hm_list3), (1, 2, 0)), np.transpose(np.array(hm_list2), (1, 2, 0)), np.transpose(np.array(hm_list), (1, 2, 0))
    #return  np.transpose(np.array(hm_list), (1, 2, 0))

def create_depth_hm(hm_2d, z_root, z):
    # hm_2d list of k hms
    # z_root, relative to this depth
    # z, all depths
    # return, list of k depth hms
    hm_2d = np.transpose(np.array(hm_2d), (2, 0, 1))
    #print(np.shape(hm_2d[0]))
    hm_depth = list()
    for i in range(21):
        z_rel = z_root - z[i]
        hm_depth_k = z_rel * hm_2d[i]
        hm_depth.append(hm_depth_k)

    return np.transpose(np.array(hm_depth), (1, 2, 0))

def create_onehot(uv, w, h):
    heats = list()
    temp_im = np.zeros((w,h,21))
    temp_im2 = np.zeros((w*2,h*2,21))
    temp_im3 = np.zeros((w*4,h*4,21))
        # img = list()
    for j,coord in enumerate(uv):
            # temp_im = np.zeros((224,224))
        try: 
            temp_im[int(coord[0]/4), int(coord[1]/4),j] = 1
            temp_im2[int(coord[0]/2), int(coord[1]/2),j] = 1
            temp_im3[int(coord[0]), int(coord[1]),j] = 1
        except:
            print("\n Coordinates where out of range : " , coord[0], coord[1])
    return temp_im, temp_im2, temp_im3


def get_depth(xyz_list):
    depth = np.zeros(21)
    xyz = np.array(xyz_list)
    for j in range(21):
        depth[j] = xyz[j, 2]
    return depth

def get_data(dir_path, num_samples, multi_dim = True):
    print("Collecting data ... \n")
    imgs = []
    uv = []
    coords = []
    hm = []
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    n = 0
    gaussian = create_gaussian_blob(2, 3)
    for i in range(num_samples, num_samples+200):
        # load images
        img = read_img(i, dir_path, 'training')
        imgs.append(img)
        uv_i = projectPoints(xyz_list[i], K_list[i])
        uv_i = uv_i[:, ::-1]
        uv.append(uv_i)
        hm_tmp = create_gaussian_hm(uv_i,224,224, 3, gaussian)

        hm.append(hm_tmp[2])
        coords.append([])
        for j, coord in enumerate(uv_i):
            # save coordinates
            coords[n].append(coord[0])
            coords[n].append(coord[1])
        n = n+1
    return imgs, uv, np.array(hm), np.array(coords)


def get_trainData(dir_path, num_samples, multi_dim = True):
    print("Collecting data ... \n")
    imgs = []
    heats = []
    heats2 = []
    heats3 = []
    coords = []
    hm_depth = []
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    gaussian = create_gaussian_blob(2,3)
    for i in range(num_samples):
        # load images
        img = read_img(i, dir_path, 'training')
        imgs.append(img)
        # project 3d coords and create heatmaps
        uv = projectPoints(xyz_list[i], K_list[i])
        onehots = create_onehot(uv, 56, 56)
        hm = create_gaussian_hm(uv,224,224, 3, gaussian)
        heats.append(hm[2])
        heats2.append(onehots[1])
        heats3.append(onehots[2])
        coords.append([])
        z = []

        for j, coord in enumerate(uv):
            # save coordinates
            coords[i].append(coord[0])
            coords[i].append(coord[1])
            z.append(xyz_list[i][j][2])

        # tmp = np.transpose(np.array(heats[i]), (2, 0, 1))

        # for training this format should work
      #  hm_depth.append(create_depth_hm(heats[i], z[0], z))
    return np.array(imgs), np.array(heats), np.array(heats2), np.array(heats3), np.array(coords)


def get_evalImages(dir_path, num_samples):
    print("Collecting data evaluation data ... \n")
    imgs = []
    for i in range(num_samples):
        # load images
        img = read_img(i, dir_path, 'training')
        imgs.append(img)

    return np.array(imgs)



def dataGenerator(dir_path, batch_size = 16, data_set = 'training'):
   # hm_all = np.array(read_csv(dir_path+'/hm.csv'))


    if data_set == 'training':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[:-560]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[:-560]
        K_list *= 4
        indicies = [i for i in range(32000)] + [i for i in range(32560,64560)] + [i for i in range(65120,97120)] + [i for i in range(97680,129680)]
        print("Total number of training samples: ", num_samples, " and ", len(indicies))
    elif data_set == 'validation':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:]
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
        K_list *= 4
        indicies = [i for i in range(32000,32560)] + [i for i in range(64560,65120)] + [i for i in range(97120, 97680)] + [i for i in range(129680,130240)]
        print("Total number of validation samples: ", num_samples," and ", len(indicies))
    elif data_set == 'evaluation':
        xyz_list = json_load(os.path.join(dir_path, 'evaluation_xyz.json'))
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))

    else:
        print("No specified data found!")
        sys.exit()
        
    tmp = True
    i = 0

   # tic = time.perf_counter()

   # toc = time.perf_counter()
   # print(f"hm_small{toc - tic:0.4f} seconds")

    while True:
      #  tic = time.perf_counter()

        batch_x = []
       # batch_y = [[], [], [], []]
        batch_y = [[], [], []]
        #batch_y = [[]]
        for j in range(batch_size):
            idx = indicies[i+j]
           # print(idx)
            img = read_img(idx, dir_path, 'training')/255.0
           # hm2 = read_hm(idx, dir_path, 'training')/255.0
            uv = projectPoints(xyz_list[i+j], K_list[i+j])
            #depth = get_depth(xyz_list[i+j])
            #onehots = create_onehot(uv, 56, 56)
            hm = create_gaussian_hm(uv, 224, 224, 3,  hm_small)
            # z = []
           # for k in range(21):
                # save depth coordinates
               # d = xyz_list[i+j][0][2] -xyz_list[i+j][k][2]
               # d = xyz_list[i+j][k][2]
               # z.append(d) # TODO: correct indices?
               # if tmp:
                   # print(z[0]-z[k])
         #   tmp = False
            #print(np.shape(onehots[2]))
      #      depthmap = create_depth_hm(hm[2], z[0], z)#get_depthmaps(uv, xyz_list[idx])

            #print(np.shape(depthmap))
            batch_x.append(img)
            batch_y[0].append(hm[0])
            batch_y[1].append(hm[1])
            batch_y[2].append(hm[2])
          #  batch_y[3].append(depthmap)
           # batch_y[3].append(z)
            if i+j == num_samples-1:
                i = -j
        i += batch_size
        #toc = time.perf_counter()

       # print(f"one generator {toc - tic:0.4f} seconds")

        yield (np.array(batch_x), batch_y)


def dataGenerator_save_hm(dir_path, batch_size=16, data_set='training'):

    if data_set == 'training':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
        xyz_list *= 4
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))
        K_list *= 4
        indicies = [i for i in range(32560)] + [i for i in range(32560, 65120)] + [i for i in range(65120, 97680)] + [i
                                                                                                                      for
                                                                                                                      i
                                                                                                                      in
                                                                                                                      range(
                                                                                                                          97680,
                                                                                                       130240)]
        print("Total number of training samples: ", num_samples, " and ", len(indicies))
    elif data_set == 'evaluation':
        xyz_list = json_load(os.path.join(dir_path, 'evaluation_xyz.json'))
        num_samples = len(xyz_list)
        print("Total number of evaluation samples: ", num_samples)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))

    else:
        print("No specified data found!")
        sys.exit()

    tmp = True
    i = 0
    std = 2
    radius = 6
    hm_small = create_gaussian_blob(std, radius)

    while True:
        batch_x = []
        # batch_y = [[], [], [], []]
        batch_y = [[], [], []]
        for j in range(batch_size):
            idx = indicies[i + j]
         #   print(idx)
            img = read_img(idx, dir_path, 'training') / 255.0
            uv = projectPoints(xyz_list[i + j], K_list[i + j])
            uv = uv[:, ::-1]
            hm = create_gaussian_hm(uv, 224, 224, radius, hm_small)
            #write_img(hm, idx, dir_path, 'training')
            #hm2 = read_hm(idx, dir_path, 'training')/255.0
           # append_list_as_row(dir_path+'/hm.csv', hm)
            # z = []
            # for k in range(21):
            # save depth coordinates
            # d = xyz_list[i+j][0][2] -xyz_list[i+j][k][2]
            # d = xyz_list[i+j][k][2]
            # z.append(d) # TODO: correct indices?
            # if tmp:
            # print(z[0]-z[k])
            #   tmp = False
            # print(np.shape(onehots[2]))
            #      depthmap = create_depth_hm(hm[2], z[0], z)#get_depthmaps(uv, xyz_list[idx])

            batch_x.append(img)
            batch_y[0].append(hm)
            batch_y[1].append(uv)
            batch_y[2].append(hm)
            #  batch_y[3].append(depthmap)
            # batch_y[3].append(z)
            if i + j == num_samples - 1:
                i = -j
        i += batch_size

        yield (np.array(batch_x), batch_y)


def save_preds(dir_path, predictions):
    #Function that saves away the predictions as jpg images.
    save_path = dir_path + "predictions/test"
    makedirs(save_path)
    print("Saving the predictions to " + save_path)
    predictions *= 255
    for i, pred in enumerate(predictions):
        name = save_path + str(i)+".jpg"
        io.imsave(name, pred.astype(np.uint8))


def plot_predicted_heatmaps(preds, heatmaps):
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 5
    n = 0
    for i in range(1,rows+1,2):
        fig.add_subplot(rows, columns, i)
        plt.imshow(preds[0][:, :, n])
        plt.colorbar()
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(heatmaps[0][:, :, n])
        plt.colorbar()
        n += 1
    #plt.show()
    plt.savefig('heatmaps.png')
    plt.figure()
    plt.imshow(np.sum(heatmaps[0][:, :, :], axis=-1))
    plt.savefig('hand_as_hm.png')


def plot_hm_with_images(hm_pred, hm_true, img, ind=0):
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 5
    n = 0
    for i in range(1,rows*3,3):
        fig.add_subplot(rows, columns, i)
       # print(ind)
       # print(n)
        plt.imshow(hm_pred[ind][:, :, n])
        plt.colorbar()
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(hm_true[ind][:, :, n])
        ind_pred = np.unravel_index(np.argmax(hm_pred[ind][:, :, n], axis=None), hm_pred[ind][:, :, n].shape)
        ind_true = np.unravel_index(np.argmax(hm_true[ind][:, :, n], axis=None), hm_true[ind][:, :, n].shape)
        plt.scatter(ind_pred[1], ind_pred[0], c='r', s=2)
        plt.scatter(ind_true[1], ind_true[0], c='b', s=2)
       # print('-----')
       # print(ind_pred[1], ind_pred[0])

        plt.colorbar()
        fig.add_subplot(rows, columns, i + 2)
        plt.imshow(img[ind])
        #for i in range(21):
       # ind_pred = np.unravel_index(np.argmax(hm_pred[ind][:, :, n], axis=None), hm_pred[ind][:, :, n].shape)
       # ind_true = np.unravel_index(np.argmax(hm_true[ind][:, :, n], axis=None), hm_true[ind][:, :, n].shape)
        plt.scatter(ind_pred[1], ind_pred[0], c='r', s=2)
        plt.scatter(ind_true[1], ind_true[0], c='b', s=2)
      #  print(ind_pred[1], ind_pred[0])
        n += 1
    #plt.show()
    plt.savefig('extended_hm'+str(ind)+'.png')



def plot_predicted_heatmaps_tmp(preds, heatmaps):
    print(preds.shape)
    print(heatmaps.shape)
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(preds[:, :, 1])
    plt.colorbar()
    fig.add_subplot(1, 2, 1 + 1)
    plt.imshow(heatmaps[:, :, 1])
    plt.colorbar()

    plt.show()
    plt.savefig('heatmaps.png')

def plot_predicted_depthmaps(preds):
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 8
    n = 0
    coords = []
    for i in range(1,rows+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(preds[0][:, :, n])
        plt.colorbar()
        n += 1
        coords.append(np.sum(preds[0][:, :, n]))
    plt.show()
    plt.savefig('heatmaps_depth.png')
    print("Predicted depths")
    print(coords)





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
        plt.savefig('scatter.png')
        #plt.show()

    except:
        print('Error in scatter plot')



def plot_predicted_hands(images, coord_preds):
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(images[i - 1])
        # nesed to project onto image..
        one_pred = [coord_preds[i-1][0::2], coord_preds[i-1][1::2]]
        plot_hand(ax, np.transpose(np.array(one_pred)))


    #plt.show()
    plt.savefig('hands.png')



def plot_predicted_hands_uv(images, coord_preds, name):
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        try:
            plt.imshow(images[i - 1])
            # need to project onto image..
            one_pred = [coord_preds[i-1][0::2], coord_preds[i-1][1::2]]
            plot_hand(ax, np.transpose(np.array(one_pred)),order='uv')
        except:
            print('error')
            continue
    plt.savefig(name)
  #plt.show()


def add_depth_to_coords(coords, depth):
    xyz = [coords[0::2], coords[1::2], depth]
    return np.array(xyz).T




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
    # ax.view_init(elev=-85, azim=-75)
    # pickle.dump(fig, open('3DHands.fig.pickle', 'wb'))
    plt.savefig('hands_3D.png')

    # ret = fig2data(fig)  # H x W x 4
    plt.close(fig)
    # return ret

# def fig2data(fig):
#     """
#     @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
#     @param fig a matplotlib figure
#     @return a numpy 3D array of RGBA values
#     """
#     # draw the renderer
#     fig.canvas.draw()

#     # Get the RGBA buffer from the figure
#     w, h = fig.canvas.get_width_height()
#     buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
#     buf.shape = (w, h, 4)

#     # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#     buf = np.roll(buf, 3, axis=2)
#     return buf
