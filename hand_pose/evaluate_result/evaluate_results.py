import getopt
import os
import sys
sys.path.insert(1, '../')

import numpy as np
from evaluate_result.evaluate import evaluate_result
import matplotlib.pyplot as plt

from utils.help_functions import get_evalImages, get_raw_data
import matplotlib
from utils.fh_utils import projectPoints
from utils.plot_functions import plot_hand
from utils.plot_3d_skeleton import draw_3d_skeleton


def add_z_root(z_root_p, xyz, s, K):
    '''Adding z root to 3d prediction and projects onto image'''
    xyz_abs_p = []
    uv_proj = []
    for (ind, z) in enumerate(z_root_p):
       # print(z)
        xyz_p = xyz[ind]
        xyz_add_root = np.copy(xyz_p)
        xyz_add_root[:, 2] = z + xyz_p[:, 2]
       # print(xyz_add_root)

        xyz_abs_p.append(xyz_add_root)
        uv_proj.append(np.array(projectPoints(xyz_add_root, K[ind])))

    xyz_abs_p = np.array(xyz_abs_p)

    uv_proj = np.array(uv_proj)
  #  print(xyz_abs_p)
    return xyz_abs_p, uv_proj




def calculate_zroot(xyz, uv, K_list):
    '''Calculate zroot by projecting with different values and take best'''
    z_root = []
    z_root_errors = []
    for i in range(len(xyz)):
        xyz_p = xyz[i]
        uv_p = uv[i]
        K = K_list[i]
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
        z_root.append(z[np.argmin(diff)])
        z_error = np.abs(z[np.argmin(diff)]-xyz_list[i][0,2])
        z_root_errors.append(z_error)

    return np.array(z_root), np.array(z_root_errors)

def write_in_excel_format(result_path, hm, test):
    score_path = os.path.join(result_path + 'score'+test+'/score_root_relative.txt')
    with open(score_path, 'r') as fo:
        a = fo.read()
        a = a.split('\n')
        print(a)
        xyz_mean3d = a[0].split(" ")[1]
        xyz_auc3d = a[1].split(" ")[1]
        xyz_al_mean3d = a[2].split(" ")[1]
        xyz_al_auc3d = a[3].split(" ")[1]
        xy_mean2d = a[4].split(" ")[1]
        xy_al_mean2d = a[6].split(" ")[1]
    score_path = os.path.join(result_path + 'score'+test+'/score_absolute_coordinates.txt')
    with open(score_path, 'r') as fo:
        a = fo.read()
        a = a.split('\n')
        print(a)
        a_xyz_mean3d = a[0].split(" ")[1]
        a_xyz_auc3d = a[1].split(" ")[1]
        a_xyz_al_mean3d = a[2].split(" ")[1]
        a_xyz_al_auc3d = a[3].split(" ")[1]
        a_xy_mean2d = a[4].split(" ")[1]
        a_xy_al_mean2d = a[6].split(" ")[1]
    # a = a.split(" ")
    with open(result_path + 'xyz_loss.csv', 'r') as f:
        lines = f.read().splitlines()
        xyz_loss = lines[-1]
    with open(result_path + 'val_xyz_loss.csv', 'r') as f:
        lines = f.read().splitlines()
        val_xyz_loss = lines[-1]
    with open(result_path + 'loss.csv', 'r') as f:
        lines = f.read().splitlines()
        loss = lines[-1]
    with open(result_path + 'val_loss.csv', 'r') as f:
        lines = f.read().splitlines()
        val_loss = lines[-1]
    if hm:
        with open(result_path + 'hm_loss.csv', 'r') as f:
            lines = f.read().splitlines()
            hm_loss = lines[-1]
        with open(result_path + 'val_hm_loss.csv', 'r') as f:
            lines = f.read().splitlines()
            val_hm_loss = lines[-1]
    else:
        with open(result_path + 'uv_loss.csv', 'r') as f:
            lines = f.read().splitlines()
            hm_loss = lines[-1]
        with open(result_path + 'val_uv_loss.csv', 'r') as f:
            lines = f.read().splitlines()
            val_hm_loss = lines[-1]
    score_path = os.path.join(result_path + 'score'+test+'/to_excell.txt')

    with open(score_path, 'w') as fo:

        fo.write('%s\n' % xy_mean2d)
        fo.write('%s\n' % xy_al_mean2d)
        fo.write('%s\n' % a_xy_mean2d)
        fo.write('%s\n' % a_xy_al_mean2d)
        fo.write('%s\n' % xyz_mean3d)
        fo.write('%s\n' % xyz_al_mean3d)
        fo.write('%s\n' % xyz_auc3d)
        fo.write('%s\n' % xyz_al_auc3d)
        fo.write('%s\n' % a_xyz_mean3d)
        fo.write('\n\n\n')
        fo.write('%s\n' % xyz_loss)
        fo.write('%s\n' % val_xyz_loss)
        fo.write('%s\n' % hm_loss)
        fo.write('%s\n' % val_hm_loss)
        fo.write('\n')
        fo.write('%s\n' % loss)
        fo.write('%s\n' % val_loss)


if __name__ == '__main__':
    '''Evaluating results and calculate z-root value
    score/to_excel.txt here you can just copy and insert to our excel sheet
    if hm = False: loss file should be in uv_loss.csv and val_uv_loss.csv
    if got_3d = False: Should evaluate 2D results only
    Files that should exist: xyz_pred.csv uv_pred.csv
    '''

    #Path to results
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/ED_50e_separated/'
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/ED_50e_shuffled/'
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/ED_50e_112/'
    #result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/MN_50e/'
    #result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/hm/MN_30e/'

    #num_images = 5
    #Path to dataset
#todo run this code ande re-run predictions for mobnet
    freihand_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"
    test = ""
    dataset = 'validation'
    if dataset == 'test':
        test = '_test'

    got_3d = True
    hm = True
    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:f:v:", ["path=","freihand_path=","version="])
    except getopt.GetoptError:
        print('Require inputs -e <epochs> -t <training_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-p':
            result_path = str(arg)
        elif opt == '-f':
            freihand_path = str(arg)
        elif opt == '-v':
            version = str(arg)
            if version == '2d':
                got_3d = False


    if got_3d:
        xyz_pred_file = result_path + 'xyz_pred'+test+'.csv'
    uv_pred_file = result_path + 'uv_pred'+test+'.csv'
    # Load data from freihand
    xyz_list, K_list, num_samples, s_list = get_raw_data(freihand_path, data_set=dataset)
    xyz_list = np.array(xyz_list)
    K_list = np.array(K_list)
    s_list = np.array(s_list)
    # Load results from network, scale uv to original size
    if got_3d:
        pred_xyz = np.reshape(np.loadtxt(xyz_pred_file, delimiter=','), (-1, 21, 3))
    pred_uv = 4 * np.reshape(np.loadtxt(uv_pred_file, delimiter=','), (-1, 21, 2))

    print(s_list.shape)
    print(xyz_list.shape)
    print(pred_xyz.shape)
    # Directory to save result in
    directory = ["score"+test, "images"+test, "reconstructed"+test]
    save_path = ["score"+test+"/score_root_relative","score"+test+"/score_absolute_coordinates"]

    for d in directory:
        try:
            os.mkdir(os.path.join(result_path, d))
        except OSError as error:
            print(error)
    num_images = 2027
    images = get_evalImages(freihand_path, num_images, dataset=dataset)
    show_bad_zroot = True
    if got_3d:
        # Compare relative root result
        target_xyz_rel = np.copy(xyz_list)
    # Modify labels to relative z and scale pose to original
    target_uv = []
    for i in range(num_samples):
        target_uv.append(projectPoints(xyz_list[i], K_list[i]))
        if got_3d:
            target_xyz_rel[i][:,2] = target_xyz_rel[i][:,2]-target_xyz_rel[i][0,2]
            pred_xyz[i] = pred_xyz[i] * s_list[i]
        else:
            pred_xyz = [0]
            target_xyz_rel = [0]
    matplotlib.use('TkAgg')
    fig = plt.figure()

    fig, z_lim = draw_3d_skeleton(target_xyz_rel[2026], (224 * 2, 224 * 2), subplot=True, ind=1,f=fig)

    draw_3d_skeleton(pred_xyz[2026], (224 * 2, 224 * 2), subplot=True, ind=2, f=fig, z_lim=z_lim)
    fig = plt.figure()
    images=np.array(images)
    print(images.shape)
    plt.imshow(images[2026])
    plt.show()
    target_uv = np.array(target_uv)
    print((target_uv.shape))
    print((pred_uv.shape))
    evaluate_result(result_path, save_path[0], pred_xyz, target_xyz_rel, pred_uv, target_uv[:len(pred_uv)], got_3d=got_3d)

    # Calculate relative root and project onto 2D
    # Only if not done before..
    pred_xyz_abs = []

    if got_3d:
        try:
            pred_xyz_abs = np.reshape(np.loadtxt(result_path+'reconstructed'+test+'/xyz_pred_abs.csv',delimiter=','), (-1,21,3))
            pred_uv_abs = np.reshape(np.loadtxt(result_path + 'reconstructed'+test+'/uv_pred_abs.csv', delimiter=','), (-1, 21, 2))
            z_root_errors = np.loadtxt(result_path + 'reconstructed'+test+'/z_root_errors.csv', delimiter=',')
        except:
            uv_pred = []
            z_root_errors =[]
            #TODO: Is this the best way? back project 2d could be an option
            z_root, z_root_errors = calculate_zroot(pred_xyz[:len(pred_uv)], pred_uv, K_list)
            pred_xyz_abs, pred_uv_abs = add_z_root(z_root, pred_xyz, s_list, K_list)
            np.savetxt(result_path+'reconstructed'+test+'/xyz_pred_abs.csv', np.reshape(pred_xyz_abs, (-1,63)), delimiter=',')
            np.savetxt(result_path+'reconstructed'+test+'/uv_pred_abs.csv', np.reshape(pred_uv_abs,(-1,42)), delimiter=',')
            np.savetxt(result_path+'reconstructed'+test+'/z_root_errors.csv', z_root_errors, delimiter=',')
        print(len(pred_xyz_abs))
        print(len(xyz_list))
        evaluate_result(result_path, save_path[1], pred_xyz_abs, xyz_list[:len(pred_xyz_abs)], pred_uv_abs, target_uv[:len(pred_uv_abs)], got_3d=got_3d)
    if got_3d:
        score_path = os.path.join(result_path+'score'+test+'/z_root_error.txt')
        calc_bad = np.sum(z_root_errors>0.1)
        z_root_errors = np.array(z_root_errors)
        with open(score_path, 'w') as fo:
            fo.write('Number of z roots off by more than 10 cm: %d of %d, %d percent\n' % (calc_bad,num_samples, (calc_bad/num_samples)))
            fo.write('Mean of error of zroot: %f \n' % np.mean(z_root_errors))
            fo.write('Median of error of zroot: %f\n' % np.median(z_root_errors))
            fo.write('Std of error of zroot: %f\n' % np.std(z_root_errors))
        #matplotlib.use('TkAgg')
        plt.figure()
        plt.hist(z_root_errors)
        plt.title('Distribution of z-root error')
        plt.savefig(result_path+'images'+test+'/z_error_dist.png')
        # print to file in specific order..
        write_in_excel_format(result_path, hm, test)

    # KÖR INTE?





