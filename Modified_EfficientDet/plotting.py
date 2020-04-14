from numpy import loadtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import sys
import os
from FreiHAND.freihand_utils import json_load

color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little



def draw_3d_skeleton(pose_cam_xyz,target_xyz,image, image_size):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(131, projection='3d')
    marker_sz = 15
    line_wd = 2

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 2],
                -pose_cam_xyz[joint_ind:joint_ind + 1, 1], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 2], -pose_cam_xyz[[0, joint_ind], 1],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 2],
                    -pose_cam_xyz[[joint_ind - 1, joint_ind], 1], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)


    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    xmargin = 0.3
    ymargin = 0.3
    zmargin = 0.3
    ax.margins(x=xmargin, y=ymargin, z=zmargin)


    ax2 = plt.subplot(132, projection='3d')
    for joint_ind in range(target_xyz.shape[0]):
        ax2.plot(target_xyz[joint_ind:joint_ind + 1, 0], target_xyz[joint_ind:joint_ind + 1, 2],
                -target_xyz[joint_ind:joint_ind + 1, 1], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax2.plot(target_xyz[[0, joint_ind], 0], target_xyz[[0, joint_ind], 2], -target_xyz[[0, joint_ind], 1],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax2.plot(target_xyz[[joint_ind - 1, joint_ind], 0], target_xyz[[joint_ind - 1, joint_ind], 2],
                    -target_xyz[[joint_ind - 1, joint_ind], 1], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)


    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('Y')
    ax2.margins(x=xmargin, y=ymargin, z=zmargin)

    ax3 = plt.subplot(133)
    ax3.imshow(image)

    # pickle.dump(fig, open('3DHands.fig.pickle', 'wb'))
    plt.savefig('hands_3D.png')
    plt.show()

import numpy as np
from help_functions import read_img

image_nbr = 0
dir_path = '..\..\data\FreiHAND_pub_v2'
xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:]
xyz_target = np.array(xyz_list[image_nbr])
# print(xyz_target)     
data = loadtxt('pose_cam_xyz.csv', delimiter=',')
print(np.shape(data))
print(data[:21])
data = np.array(data).reshape(-1,21,3)
print(data[0])
indicies = [i for i in range(32000,32560)]
# print(data[image_nbr])
# image = pickle.load(open('hand_for_3d.fig.pickle', 'rb'))
image = read_img(indicies[image_nbr],dir_path,'training')
draw_3d_skeleton(data[image_nbr],xyz_target, image,image_size= (224*2,224*2))
