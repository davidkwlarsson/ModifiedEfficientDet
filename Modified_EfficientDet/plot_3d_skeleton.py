import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from help_functions import get_evalImages
from utils.fh_utils import *
from FreiHAND.tfdatagen_frei import get_raw_data


color_hand_joints =   [[1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # thumb
                       [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # index
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
   #plt.grid(b=None)
    #plt.axis('off')

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
 #   ax.set_zlim((0.53,0.8))
    ax.set_xlim(-0.08, 0.08)
    ax.set_ylim(-0.08, 0.08)
    ax.set_zlim(-0.08, 0.08)


   # ax.set_zlim(0.6, 0.8)
# ax.view_init(elev=-85, azim=-75)
    ax.view_init(elev=0, azim=0)
    # pickle.dump(fig, open('3DHands.fig.pickle', 'wb'))
    plt.savefig('hands_3D.png')
    # ret = fig2data(fig)  # H x W x 4
    #plt.close(fig)
    # return ret

if __name__ == '__main__':
    scale = True
    add_root = False
    dir_path = "../../data/FreiHAND_pub_v2/"  #
    images = get_evalImages(dir_path, 10, dataset='validation')
    #path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e_fewer_weights/'
    #path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e_fw_no_es/'
    path = ""
    xyz_list, K_list, num_samples, s_list = get_raw_data(dir_path, 'validation')
 #   print(np.shape(images))
    uv_t = np.loadtxt(path+'uv_targets2.csv', delimiter=',')*4
    uv = np.loadtxt(path+'uv_preds2.csv', delimiter=',')*4
    im = []
    for i in images:
        im.append(cv2.resize(i,(224,224)))

   # images = im
    pred_coords = np.loadtxt(path+'pose_cam_xyz_pred_.csv', delimiter=',')
    target_coords_t = np.loadtxt(path+'pose_cam_xyz_target_.csv', delimiter=',')
    print('pred_coords',pred_coords.shape)
    print('target_coords_t',target_coords_t.shape)
   # n = pred_coords.shape[0]/21
    #print(n)
    pred_coords = np.reshape(pred_coords,(-1,21,3))
    target_coords_t = np.reshape(target_coords_t,(-1,21,3))
    for i in range(10):
        if scale:
            s = s_list[i]
        else:
            s=1
       # coords = np.loadtxt('pose_cam_xyz_pred_'+str(i)+'.csv', delimiter=',')
       # coords_t = np.loadtxt('pose_cam_xyz_target_'+str(i)+'.csv', delimiter=',')
        coords_t = target_coords_t[i]
        coords = pred_coords[i]
        if add_root:
            xyz = np.array(xyz_list[i])
            print(xyz.shape)
            coords_t = coords_t*s
            coords = coords*s
            coords[:,2] = coords[:,2] + xyz[0,2]
            coords_t[:,2] = coords_t[:,2] + xyz[0,2]
        else:
            coords_t = coords_t*s
            coords = coords*s
            #coords[:, 2] = coords[:, 2] * s
            #coords_t[:, 2] = coords_t[:, 2] * s
        print('coords shape', coords.shape)
        draw_3d_skeleton(coords, (224*2, 224*2))
        draw_3d_skeleton(coords_t, (224*2, 224*2))
        plt.figure()
        plt.imshow(images[i])
        #plt.scatter(uv[i][0::2], uv[i][1::2], marker='o', s=2, label='predicted')
        plt.scatter(uv_t[i][0::2], uv_t[i][1::2], marker='x', s=2,label='true')
        #one_pred = [uv[i][0::2], uv[i][1::2]]
        one_pred = [uv[i][0::2], uv[i][1::2]]
    # plot_hand(plt, np.transpose(np.array(one_pred)))
        plot_hand(plt, np.transpose(np.array(one_pred)), order='uv')
        #plt.legend()

        # print(coords.shape)
        plt.show()

    print('draw_3d_skeleton done')