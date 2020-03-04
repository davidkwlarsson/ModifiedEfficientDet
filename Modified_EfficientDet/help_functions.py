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
import cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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
    print(coords[0][0:2])
    plt.show()


def plot_acc_loss(history):
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
    print(coords)
    #print('coords')
    return np.array(coords)


def heatmaps_to_depth(heatmaps):
    depths = [[] for x in range(len(heatmaps))]
    for j in range(len(heatmaps)):
        for i in range(21):
            depths[j].append(np.sum(heatmaps[j][:, :, i]))

    return np.array(depths)

def get_depthmaps(uv,xyz_list):
    depths = np.zeros((224,224,21))
    xyz = np.array(xyz_list)
    for j,coord in enumerate(uv):
        try:
            depths[int(coord[0]),int(coord[1]),j] = xyz[j, 2]
        except:
            print("\n")

    return depths

# NOTE that they are not symmetric atm..
# check by running test
def create_gaussian_hm(uv, w, h):
    hm_list = list()
    hm_list2 = list()
    hm_list3 = list()
    py = 50
    px = 50
    im = np.zeros((w+2*px, h+2*py))
    std = 2
    radius = 6

    for coord in uv:
        u = coord[1]+px
        v = coord[0]+py
        hm_small = np.zeros((radius * 2 + 1, radius * 2 + 1))
        xc, yc = (radius + 1, radius + 1)
        for x in range(radius * 2 + 1):
            for y in range(radius * 2 + 1):
                dist = math.sqrt((x - xc) ** 2 + (y - yc) ** 2)
                if dist < radius:
                    scale = 1  # otherwise predict only zeros
                    hm_small[x][y] = scale * math.exp(-dist ** 2 / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))
        m = np.max(hm_small)
        for x in range(radius * 2 + 1):
            for y in range(radius * 2 + 1):
                hm_small[x][y] = hm_small[x][y]/m
        #hm_small = hm_small/np.argmax(np.argmax(hm_small))
        # plt.imshow(hm_small)
        # plt.show()
        # print('here')
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
    return np.transpose(np.array(hm_list3), (1, 2, 0)), np.transpose(np.array(hm_list2), (1, 2, 0)), np.transpose(np.array(hm_list), (1, 2, 0))

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
    for i in range(num_samples, num_samples+200):
        # load images
        img = read_img(i, dir_path, 'training')
        imgs.append(img)
        uv_i = projectPoints(xyz_list[i], K_list[i])
        uv.append(uv_i)
        hm_tmp = create_gaussian_hm(uv_i,224,224)
        hm.append(hm_tmp[2])
        coords.append([])
        for j, coord in enumerate(uv_i):
            # save coordinates
            coords[n].append(coord[0])
            coords[n].append(coord[1])
        n=n+1
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
    for i in range(num_samples):
        # load images
        img = read_img(i, dir_path, 'training')
        imgs.append(img)
        # project 3d coords and create heatmaps
        uv = projectPoints(xyz_list[i], K_list[i])
        onehots = create_onehot(uv, 56, 56)
        hm = create_gaussian_hm(uv,224,224)
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
        img = read_img(i, dir_path, 'evaluate')
        imgs.append(img)

    return np.array(imgs)


def dataGenerator(dir_path, batch_size = 16, data_set = 'training'):
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
        print("Total number of evaluation samples: ", num_samples)
        K_list = json_load(os.path.join(dir_path, 'evaluation_K.json'))

    else:
        print("No specified data found!")
        sys.exit()
        
    tmp = True
    i = 0
    while True:
        batch_x = []
       # batch_y = [[], [], [], []]
        batch_y = [[], [], []]
        for j in range(batch_size):
            idx = indicies[i+j]
            img = read_img(idx, dir_path, 'training')/255.0
            #img = cv2.resize(img, (112,112))
            uv = projectPoints(xyz_list[i+j], K_list[i+j])
            #depth = get_depth(xyz_list[i+j])
            onehots = create_onehot(uv, 56, 56)
            hm = create_gaussian_hm(uv, 224, 224)
            if tmp:
               # plot_predicted_heatmaps_tmp(hm[0],onehots[0])
               # print(uv)
                tmp = False
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
    plt.imshow(np.sum(heatmaps[0][:, :, :], axis=-1))
    plt.savefig('hand_as_hm.png')


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
        plot_hand(ax, np.transpose(np.array(one_pred)), order='uv')


    #plt.show()
    plt.savefig('hands.png')



def plot_predicted_hands_uv(images, coord_preds):
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        try:
            plt.imshow(images[i - 1])
            # need to project onto image..
            one_pred = [coord_preds[i-1][0::2], coord_preds[i-1][1::2]]
            plot_hand(ax, np.transpose(np.array(one_pred)), order='uv')
        except:
            print('error')
            continue
    plt.savefig('hands_uv.png')
    plt.show()


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
