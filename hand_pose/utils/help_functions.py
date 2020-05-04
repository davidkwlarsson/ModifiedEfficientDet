import os
import sys
import json

import math
from skimage.transform import resize
import numpy as np
from utils.fh_utils import *
import skimage.io as io
from utils.plot_functions import *


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


def create_gaussian_blob(std, radius):
    hm_small = np.zeros((radius * 2 + 1, radius * 2 + 1))
    xc, yc = (radius, radius)
    for x in range(radius * 2 + 1):
        for y in range(radius * 2 + 1):
            dist = math.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            scale = 1  # otherwise predict only zeros
            hm_small[x][y] = scale * math.exp(-dist ** 2 / (2 * std ** 2))  # / (std * math.sqrt(2 * math.pi))
    # TODO: Here I just removed the scaling part so normalizing should not be necessary?

    return hm_small


def get_raw_data(dir_path, data_set='training'):
    try:
        dir_path = dir_path.decode('ascii')
        data_set = data_set.decode('ascii')
        print('dataset', data_set)
    except:
        print('string ok')
    # xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    # xyz_list *= 4
    # K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    # #K_list *= 4'
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    s_list = json_load(os.path.join(dir_path, 'training_scale.json'))
    num_tot_samples = len(xyz_list)
    print('train top', num_tot_samples * 0.85)
    print('val bot', num_tot_samples * 0.95)
    if data_set == 'training':
        lim_left = 0
        lim_right = int(num_tot_samples * 0.85)

        # indicies = [i for i in range(32000)] + [i for i in range(32560,64560)] + [i for i in range(65120,97120)] + [i for i in range(97680,129680)]
    elif data_set == 'small_dataset':  # for testing
        lim_left = 0
        lim_right = 100

    elif data_set == 'validation':
        lim_left = int(num_tot_samples * 0.85)
        lim_right = int(num_tot_samples * 0.95)
    elif data_set == 'test':
        lim_left = int(num_tot_samples * 0.95)
        lim_right = num_tot_samples

    elif data_set == 'evaluation':
        # TODO
        xyz_list = json_load(os.path.join(dir_path, 'evaluation_xyz.json'))
        num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))
        s_list = json_load(os.path.join(dir_path, 'evaluation_scale.json'))

    else:
        print("No specified data found!")
        print('dir_path')
        sys.exit()
    xyz_list = xyz_list[lim_left:lim_right]  # training on 85% of data
    K_list = K_list[lim_left:lim_right]
    s_list = s_list[lim_left:lim_right]
    xyz_list *= 4
    K_list *= 4
    s_list *= 4
    num_samples = len(xyz_list)
    print('Dataset', data_set)
    print("Total number of samples: ", num_samples)

    return xyz_list, K_list, int(num_samples), s_list  # , indicies, num_samples


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'
        img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                    '%08d.jpg' % sample_version.map_id(idx, version))
    if set_name == 'validation':
        img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                    '%08d.jpg' % sample_version.map_id(idx, version))

    if set_name == 'training':
        img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                    '%08d.jpg' % sample_version.map_id(idx, version))
    if set_name == 'small_dataset':
        img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                    '%08d.jpg' % sample_version.map_id(idx, version))
    if set_name == 'test':
        img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                    '%08d.jpg' % sample_version.map_id(idx, version))

    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)


def heatmaps_to_coord(heatmaps):
    """ Take max of heatmap and return its coordinates"""
    print('in hm to coord')
    coords = [[] for x in range(len(heatmaps))]
    print('len of heatmaps', len(heatmaps))
    print('shape of inner heatmaps', np.shape(heatmaps[0]))
    for j in range(len(heatmaps)):
        for i in range(21):
            # print(heatmaps[j][:, :, i])
            m = np.argmax(heatmaps[j][:, :, i], axis=None)
            ind = np.unravel_index(m, np.shape(heatmaps[j][:, :, i]))
            coords[j].append(ind[1])
            coords[j].append(ind[0])

    return np.array(coords)


def create_gaussian_hm(uv, w, h, radius, hm_small):
    hm_list = list()
    hm_list2 = list()
    hm_list3 = list()
    py = 10
    px = 10
    im = np.zeros((w + 2 * px, h + 2 * py))
    for coord in uv:
        u = coord[0] + px
        v = coord[1] + py
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

        hm_list.append(im)
        hm_list2.append(resize(im, (w / 2, h / 2)))
        hm_list3.append(resize(im, (w / 4, h / 4)))
        im = np.zeros((w + 2 * px, h + 2 * py))

    # plt.imshow(im)
    # plt.show()
    # return np.array(hm_list)
    return np.transpose(np.array(hm_list3), (1, 2, 0)), np.transpose(np.array(hm_list2), (1, 2, 0)), np.transpose(
        np.array(hm_list), (1, 2, 0))
    # return  np.transpose(np.array(hm_list), (1, 2, 0))


def create_depth_hm(hm_2d, z_root, z):
    # hm_2d list of k hms
    # z_root, relative to this depth
    # z, all depths
    # return, list of k depth hms
    hm_2d = np.transpose(np.array(hm_2d), (2, 0, 1))
    # print(np.shape(hm_2d[0]))
    hm_depth = list()
    for i in range(21):
        z_rel = z_root - z[i]
        hm_depth_k = z_rel * hm_2d[i]
        hm_depth.append(hm_depth_k)

    return np.transpose(np.array(hm_depth), (1, 2, 0))


def create_onehot(uv, w, h):
    heats = list()
    temp_im = np.zeros((w, h, 21))
    temp_im2 = np.zeros((w * 2, h * 2, 21))
    temp_im3 = np.zeros((w * 4, h * 4, 21))
    # img = list()
    for j, coord in enumerate(uv):
        # temp_im = np.zeros((224,224))
        try:
            temp_im[int(coord[0] / 4), int(coord[1] / 4), j] = 1
            temp_im2[int(coord[0] / 2), int(coord[1] / 2), j] = 1
            temp_im3[int(coord[0]), int(coord[1]), j] = 1
        except:
            print("\n Coordinates where out of range : ", coord[0], coord[1])
    return temp_im, temp_im2, temp_im3


def get_depth(xyz_list):
    depth = np.zeros(21)
    xyz = np.array(xyz_list)
    for j in range(21):
        depth[j] = xyz[j, 2]
    return depth


def get_data(dir_path, num_samples, multi_dim=True):
    print("Collecting data ... \n")
    imgs = []
    uv = []
    coords = []
    hm = []
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    n = 0
    gaussian = create_gaussian_blob(2, 3)
    for i in range(num_samples, num_samples + 200):
        # load images
        img = read_img(i, dir_path, 'training')
        imgs.append(img)
        uv_i = projectPoints(xyz_list[i], K_list[i])
        uv_i = uv_i[:, ::-1]
        uv.append(uv_i)
        hm_tmp = create_gaussian_hm(uv_i, 224, 224, 3, gaussian)

        hm.append(hm_tmp[2])
        coords.append([])
        for j, coord in enumerate(uv_i):
            # save coordinates
            coords[n].append(coord[0])
            coords[n].append(coord[1])
        n = n + 1
    return imgs, uv, np.array(hm), np.array(coords)


def get_index_data(dir_path, data_set='training'):
    if data_set == 'training':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))  # [:-560]
        xyz_list *= 4
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))  # [:-560]
        K_list *= 4
        indicies = [i for i in range(32000)] + [i for i in range(32560, 64560)] + [i for i in range(65120, 97120)] + [i
                                                                                                                      for
                                                                                                                      i
                                                                                                                      in
                                                                                                                      range(
                                                                                                                          97680,
                                                                                                                          129680)]
        num_samples = len(indicies)
        print("Total number of training samples: ", num_samples, " and ", len(indicies))
    elif data_set == 'validation':
        xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))  # [-560:]
        xyz_list *= 4
        # num_samples = len(xyz_list)
        K_list = json_load(os.path.join(dir_path, 'training_K.json'))  # [-560:]
        K_list *= 4
        indicies = [i for i in range(32000, 32560)] + [i for i in range(64560, 65120)] + [i for i in
                                                                                          range(97120, 97680)] + [i for
                                                                                                                  i in
                                                                                                                  range(
                                                                                                                      129680,
                                                                                                                      130240)]
        num_samples = len(indicies)
        print("Total number of validation samples: ", num_samples, " and ", len(indicies))
    elif data_set == 'evaluation':
        xyz_list = json_load(os.path.join(dir_path, 'evaluation_xyz.json'))
        num_samples = len(xyz_list)
        print("Total number of evaluation samples: ", num_samples)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))

    else:
        print("No specified data found!")
        sys.exit()


def get_trainData(dir_path, num_samples, multi_dim=True):
    print("Collecting data ... \n")
    imgs = []
    heats = []
    heats2 = []
    heats3 = []
    coords = []
    hm_depth = []
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))
    gaussian = create_gaussian_blob(2, 3)
    for i in range(num_samples):
        # load images
        img = read_img(i, dir_path, 'training')
        imgs.append(img)
        # project 3d coords and create heatmaps
        uv = projectPoints(xyz_list[i], K_list[i])
        onehots = create_onehot(uv, 56, 56)
        hm = create_gaussian_hm(uv, 224, 224, 3, gaussian)
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


def get_evalImages(dir_path, num_samples, dataset='training'):
    print("Collecting image data ...\n")
    imgs = []
    j = 0
    indicies = []
    total_samples = 130240
    train_samples = int(total_samples / 20 * 17)
    validation_samples = int(total_samples / 10)
    test_samples = int(total_samples / 20)
    val_part = int(validation_samples / 4)
    test_part = int(test_samples / 4)
    train_part = int(train_samples / 4)
    total_part = int(total_samples / 4)
    if dataset == 'training':
        indicies = [i for i in range(train_part)] + \
                   [i for i in range(total_part, total_part + train_part)] + \
                   [i for i in range(total_part * 2, total_part * 2 + train_part)] + \
                   [i for i in range(total_part * 3, total_part * 3 + train_part)]

    if dataset == 'validation':
        indicies = [i for i in range(train_part, train_part + val_part)] + \
                   [i for i in range(total_part + train_part, total_part + train_part + val_part)] + \
                   [i for i in range(total_part * 2 + train_part, total_part * 2 + train_part + val_part)] + \
                   [i for i in range(total_part * 3 + train_part, total_part * 3 + train_part + val_part)]
    if dataset == 'test':
        indicies = [i for i in range(train_part + val_part, train_part + val_part + test_part)] + \
                   [i for i in
                    range(total_part + train_part + val_part, total_part + train_part + val_part + test_part)] + \
                   [i for i in
                    range(total_part * 2 + train_part + val_part, total_part * 2 + train_part + val_part + test_part)] + \
                   [i for i in
                    range(total_part * 3 + train_part + val_part, total_part * 3 + train_part + val_part + test_part)]
    print('loading ', len(indicies[:num_samples]))
    indicies = indicies[:num_samples]
    for i in indicies:
        # load images
        img = read_img(i, dir_path, dataset)
        imgs.append(img)

    return np.array(imgs)


def add_depth_to_coords(coords, depth, K_list):
    xyz = []
    ## Transform the x,y coordinates back to 3D using the intrinsic camera parameters, K

    for i in range(len(coords)):
        K = np.array(K_list[i])
        x_coords = coords[i][0::2] * depth[i]
        y_coords = coords[i][1::2] * depth[i]
        uv_z = np.array([x_coords, y_coords, depth[i]])
        # print(uv_z.shape, K.shape)
        xyz.append(np.array(np.linalg.solve(K, uv_z)).T)

    return xyz

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
