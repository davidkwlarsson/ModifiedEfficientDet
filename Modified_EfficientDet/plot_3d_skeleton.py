import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from help_functions import get_evalImages
from utils.fh_utils import *

color_hand_joints =   [[1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # thumb

                       [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0], # index

                       [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # middles
                       [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                       [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little



def draw_3d_skeleton(pose_cam_xyz, image_size):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21
    matplotlib.use('TkAgg')

    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(111, projection='3d')
    marker_sz = 15
    line_wd = 2

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                np.abs(pose_cam_xyz[joint_ind:joint_ind + 1, 2]), '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], np.abs(pose_cam_xyz[[0, joint_ind], 2]),
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    np.abs(pose_cam_xyz[[joint_ind - 1, joint_ind], 2]), color=color_hand_joints[joint_ind],
                    linewidth=line_wd)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
 #   ax.set_zlim((0.53,0.8))

  #  ax.view_init(elev=-85, azim=-75)
    ax.view_init(elev=0, azim=0)
    # pickle.dump(fig, open('3DHands.fig.pickle', 'wb'))
    plt.savefig('hands_3D.png')
    # ret = fig2data(fig)  # H x W x 4
    #plt.close(fig)
    # return ret

if __name__ == '__main__':
    dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"  #
    images = get_evalImages(dir_path, 10, dataset='validation')
 #   print(np.shape(images))
    uv_t = np.loadtxt('uv_targets.csv', delimiter=',')*4
    uv = np.loadtxt('uv_preds.csv', delimiter=',')*4
    for i in range(10):
        coords = np.loadtxt('pose_cam_xyz_pred_'+str(i)+'.csv', delimiter=',')
        coords_t = np.loadtxt('pose_cam_xyz_target_'+str(i)+'.csv', delimiter=',')
        print(np.shape(uv_t))
      #  print(coords_t)
        draw_3d_skeleton(coords, (224*2, 224*2))
        draw_3d_skeleton(coords_t, (224*2, 224*2))
        plt.figure()
        plt.imshow(images[i])
        plt.scatter(uv[i][0::2], uv[i][1::2], marker='o', s=2, label='predicted')
        plt.scatter(uv_t[i][0::2], uv_t[i][1::2], marker='x', s=2,label='true')
        one_pred = [uv[i][0::2], uv[i][1::2]]
       # plot_hand(plt, np.transpose(np.array(one_pred)))
        plot_hand(plt, np.transpose(np.array(one_pred)), order='uv')
        plt.legend()
        # print(coords.shape)
        plt.show()

    print('draw_3d_skeleton done')
