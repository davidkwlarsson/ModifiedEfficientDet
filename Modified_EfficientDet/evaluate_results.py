import os

import numpy as np
from data_generators import get_raw_data
from generator import projectPoints
from evaluate import evaluate_result
import matplotlib.pyplot as plt

from help_functions import get_evalImages
import matplotlib
from utils.fh_utils import plot_hand

from plot_3d_skeleton import draw_3d_skeleton


def plot_predictions(plot,xyz_abs, xyz, xyz_p, uv, uv_p, K, s,ind):
    z_root_t = xyz_abs[0, 2]
    if plot:
        plt.plot(np.array(z)[10:], np.array(diff)[10:])
        plt.axvline(z_root_t, color='r', label='correct zroot')
        plt.axvline(z_root_p, color='b', label='predicted zroot')
        plt.legend()

    xyz_add_root = np.copy(xyz_p)
    xyz_add_root[:, 2] = z_root_p + xyz_p[:, 2]
    xyz_true_root = np.copy(xyz_p)
    xyz_true_root[:, 2] = z_root_t + xyz_p[:, 2]
    print('Z pred', z_root_p * s)
    print('Z true', z_root_t * s)
    uv_calc_root = np.array(projectPoints(xyz_add_root, K))
    uv_true_root = np.array(projectPoints(xyz_true_root, K))
    plt.figure()
    pred_uv = [uv_p[:, 0], uv_p[:, 1]]

    pred_skeleton = [uv_calc_root[:, 0], uv_calc_root[:, 1]]
    correct_skeleton = [uv[:, 0], uv[:, 1]]
    if plot:
        try:
            dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"

            images = get_evalImages(dir_path, 10, dataset='validation')
            plt.imshow(images[ind])
            plt.scatter(uv_true_root[:, 0], uv_true_root[:, 1], marker='x', s=10, label='true root val')
            plt.scatter(uv_p[:, 0], uv_p[:, 1], marker='+', s=10, label='predicted uv')
            plt.scatter(uv_c[:, 0], uv_c[:, 1], marker='o', s=10, label='true position')
            # plt.scatter(uv[:,0],uv[:,1], marker='o')
            plot_hand(plt, np.transpose(np.array(pred_uv)), order='uv')
            # plot_hand(plt, np.transpose(np.array(pred_skeleton)), order='uv')
            plt.legend()

            draw_3d_skeleton(xyz_calc_root * s, (224 * 2, 224 * 2))
            draw_3d_skeleton(xyz_abs * s, (224 * 2, 224 * 2))
            plt.show()

        except:
            print('failed')
    return uv_calc_root, xyz_calc_root, xyz_abs


def plot_(image, xyz_p, xyz_t, uv_p, uv_t, uv_network):
    pred_skeleton = [uv_p[:, 0], uv_p[:, 1]]

    matplotlib.use('TkAgg')
    draw_3d_skeleton(xyz_p, (224 * 2, 224 * 2))
    plt.title('Predicted')
    draw_3d_skeleton(xyz_t, (224 * 2, 224 * 2))
    plt.title('True')

    plt.figure()
    plt.imshow(image)
   # plt.scatter(uv_p[:, 0], uv_p[:, 1], marker='+', s=10, label='predicted uv')
    plt.scatter(uv_t[:, 0], uv_t[:, 1], marker='o', s=10, label='target uv')
    plt.scatter(uv_network[:, 0], uv_network[:, 1], marker='o', s=10, label='predicted uv')
    plot_hand(plt, np.transpose(np.array(pred_skeleton)), order='uv')
    plt.legend()
    plt.show()

def calculate_zroot(xyz_p, uv_p, K, xyz, ind):
    '''Calculate zroot by projecting with different values and take best'''

    diff = []
    z = []
    n = 0.3
    for i in range(1,600,1):
       # m.append([])
       # std.append([])
        xyz_tmp = np.copy(xyz_p)
        xyz_tmp[:, 2] = n + xyz_p[:, 2]
        uv_tmp = np.array(projectPoints(xyz_tmp, K))
        z.append(n)
        n += 0.001
       # check how good 2D projection is
        diff.append(np.sum(np.linalg.norm(uv_tmp - uv_p, axis=1)))
    z_root = z[np.argmin(diff)]

    return z_root


if __name__ == '__main__':
    #Path to dataset
    freihand_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"
    #Path to results
    #result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e_hm_feat_seperated/'
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e/'
    #result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e_fewer_weights/'
    xyz_pred_file = 'pose_cam_xyz_pred_.csv'
    uv_pred_file = 'uv_preds2.csv'
    # Directory
    directory1 = "score/score_root_relative"
    directory2 = "score/score_absolute_coordinates"
    directory3 = "score/"

    # Parent Directory path
   # result_path = "/home/User/Documents"
    # Path
    save_path1 = os.path.join(result_path, directory1)
    save_path2 = os.path.join(result_path, directory2)
    save_path3 = os.path.join(result_path, directory3)
    try:
        os.mkdir(save_path3)
    except OSError as error:
        print(error)
    images = get_evalImages(freihand_path, 10, dataset='validation')
    show_bad_zroot = False
    #Load data from freihand
    xyz_list, K_list, num_samples, s_list = get_raw_data(freihand_path, data_set='validation')
    # TODO: modify num_samples
    num_samples = 100
    xyz_list = np.array(xyz_list)[:num_samples]
    K_list = np.array(K_list)
    s_list = np.array(s_list)
    #Load results from network
    file = result_path+xyz_pred_file
    pred_xyz_coords = np.reshape(np.loadtxt(file, delimiter=','), (-1, 21, 3))
    file = result_path + uv_pred_file
    pred_xy_coords =4* np.reshape(np.loadtxt(file, delimiter=','), (-1, 21, 2))

    # Compare relative root result
    xyz_norm_rel = np.copy(xyz_list)
    # Modify labels and scale pose to original size
    uv_target = []
    for i in range(num_samples):
        uv_target.append(projectPoints(xyz_list[i], K_list[i]))
        xyz_norm_rel[i] = xyz_norm_rel[i]
        xyz_norm_rel[i][:,2]= xyz_norm_rel[i][:,2]-xyz_norm_rel[i][0,2]
        pred_xyz_coords[i] = pred_xyz_coords[i] * s_list[i]

    uv_target = np.array(uv_target)

    try:
        os.mkdir(save_path1)
    except OSError as error:
        print(error)
    evaluate_result(save_path1, pred_xyz_coords, xyz_norm_rel, pred_xy_coords, uv_target)

    # Calculate relative root and project onto 2D
    uv_pred = []
    calc_bad = 0
    z_root_errors =[]
    for i in range(num_samples):
        # add calculated zroot
        z_root = calculate_zroot(pred_xyz_coords[i], pred_xy_coords[i], K_list[i], xyz_list[i], i)
        pred_xyz_coords[i][:,2] = pred_xyz_coords[i][:,2] + z_root
        z_error = np.abs(z_root-xyz_list[i][0,2])
        z_root_errors.append(z_error)
        # project pose onto 2D
        uv_pred.append(np.array(projectPoints(pred_xyz_coords[i], K_list[i])))
        if z_error > 0.1:
            calc_bad += 1
            if show_bad_zroot:
                print(z_error)
                plot_(images[i], pred_xyz_coords[i],xyz_list[i], uv_pred[i], uv_target[i], pred_xy_coords[i])
    print(calc_bad)
    uv_pred = np.array(uv_pred)
    try:
        os.mkdir(save_path2)
    except OSError as error:
        print(error)

    evaluate_result(save_path2, pred_xyz_coords, xyz_list, uv_pred, uv_target)
    score_path = os.path.join(save_path3, 'info.txt')
    with open(score_path, 'w') as fo:
        fo.write('Number of z roots off by more than 10 cm: %d of %d, %d percent\n' % (calc_bad,num_samples, (calc_bad/num_samples)))
        fo.write('Mean of error of zroot: %f \n' % np.mean(z_root_errors))
        fo.write('Median of error of zroot: %f\n' % np.median(z_root_errors))
        fo.write('Std of error of zroot: %f\n' % np.std(z_root_errors))
    matplotlib.use('TkAgg')
    plt.hist(z_root_errors)
    plt.title('Distribution of z-root error')
    plt.savefig(save_path3+'z_error_dist.png')

