import getopt
import os
import sys


sys.path.insert(1, '../')

import matplotlib.pyplot as plt
import numpy as np
from utils.fh_utils import plot_hand
from utils.help_functions import get_raw_data, get_evalImages, projectPoints
from utils.plot_3d_skeleton import draw_3d_skeleton

def plot_z_root_options():
    plt.plot(np.array(z)[10:], np.array(diff)[10:])
    plt.axvline(z_root_t, color='r', label='correct zroot')
    plt.axvline(z_root_p, color='b', label='predicted zroot')
    plt.legend()

def plot_hm_with_images(hm_pred, uv_true, uv_pred, img, path, ind):
    # fig = plt.figure()
    columns = 5
    rows = 2
    n = 0
    fig, axs = plt.subplots(rows, columns, figsize=(10, 3))

    for i in range(1, columns * 2, 2):

        axs[0, n].imshow(hm_pred[:, :, n])

        axs[0, n].scatter(uv_true[n, 0] / 4, uv_true[n, 1] / 4, c='r', s=2, label='true')
        axs[0, n].scatter(uv_pred[n, 0], uv_pred[n, 1], c='b', s=2, label='predicted')

        axs[1, n].imshow(img)

        axs[1, n].scatter(uv_true[n, 0], uv_true[n, 1], c='r', s=2)
        axs[1, n].scatter(4 * uv_pred[n, 0], 4 * uv_pred[n, 1], c='b', s=2)
        n += 1
    axs[0, n-1].legend(loc="upper right",fontsize='x-small', framealpha=0.3, borderpad=0.1)

    fig.suptitle('Thumb')

    # plt.show()
    plt.savefig(path + 'images/hm_' + str(ind) + '.png')
    plt.close(fig)


def plot_skeleton_all(uv_preds,uv_preds_proj, uv_trues, images, path):
    columns = 5
    rows = 3
    n = 0
    fig, axs = plt.subplots(rows, columns, figsize=(10, 8))
    print(uv_preds.shape)
    for i in range(1, columns * 2, 2):
        pred_skeleton = [4 * uv_preds[n][:, 0], 4 * uv_preds[n][:, 1]]
        proj_skeleton = [uv_preds_proj[n][:, 0], uv_preds_proj[n][:, 1]]
        true_skeleton = [uv_trues[n][:, 0], uv_trues[n][:, 1]]

        axs[0, n].imshow(images[n])
        axs[0, n].axis('off')

        plot_hand(axs[0, n], np.transpose(np.array(pred_skeleton)), order='uv')

        axs[1, n].imshow(images[n])
        axs[1, n].axis('off')
        plot_hand(axs[1, n], np.transpose(np.array(proj_skeleton)), order='uv')

        axs[2, n].imshow(images[n])
        axs[2, n].axis('off')
        plot_hand(axs[2, n], np.transpose(np.array(true_skeleton)), order='uv')
        if n == 2:
            axs[0, n].set_title('Predictions')
            axs[1, n].set_title('Predictions, projected 3D')
            axs[2, n].set_title('True')

        n += 1
    fig.suptitle('Skeleton pose')

    plt.savefig(path)
   # plt.show()
    plt.close(fig)

def plot_skeleton(uv_preds, uv_trues, images, path):
    columns = 5
    rows = 2
    n = 0
    fig, axs = plt.subplots(rows, columns, figsize=(10, 3))
    print(uv_preds.shape)
    for i in range(1, columns * 2, 2):
        pred_skeleton = [4 * uv_preds[n][:, 0], 4 * uv_preds[n][:, 1]]
        true_skeleton = [uv_trues[n][:, 0], uv_trues[n][:, 1]]

        axs[0, n].imshow(images[n])
        axs[0, n].axis('off')

        plot_hand(axs[0, n], np.transpose(np.array(pred_skeleton)), order='uv')

        axs[1, n].imshow(images[n])
        axs[1, n].axis('off')

        plot_hand(axs[1, n], np.transpose(np.array(true_skeleton)), order='uv')
        if n == 2:
            axs[0, n].set_title('Predictions')
            axs[1, n].set_title('True')

        n += 1
    fig.suptitle('Skeleton pose')
    plt.savefig(path)
   # plt.show()
    plt.close(fig)


def plot_3d(pred_xyz, target_xyz, path,s,  visualize_3d=False, reconstructed=False, test=''):
    if reconstructed:
        save_name = path + 'images'+test+'/3d_abs_'
    else:
        save_name = path + 'images'+test+'/3d_'

    for i in range(len(pred_xyz)):
        fig = plt.figure()
        fig, z_lim = draw_3d_skeleton(pred_xyz[i] * s[i], (224 * 2, 224 * 2), subplot=True, ind=1, f=fig)
        #plt.savefig(path + 'images/3d_pred_'+str(i)+'.png')
        draw_3d_skeleton(target_xyz[i], (224 * 2, 224 * 2), subplot=True, ind=2, f=fig, z_lim=z_lim)
        plt.savefig(save_name+str(i)+'.png')
        if visualize_3d:
            plt.show()

        plt.close(fig)


def plot_3d_multiple(pred_xyz, target_xyz, path,s, images, visualize_3d=False, reconstructed=False, test=''):
    fig = plt.figure(figsize=(8,10))
    row = len(pred_xyz)
    col = 3
    ind = 1
    if reconstructed:
        save_name = path + 'images'+test+'/3d_abs.png'
    else:
        save_name = path + 'images'+test+'/3d.png'

    for i in range(row):
        fig, z_lim = draw_3d_skeleton(pred_xyz[i] * s[i], (224 * 2, 224 * 2), subplot=True, ind=ind,
                                      f=fig, subplot_size=(row,col))
        ind += 1
        draw_3d_skeleton(target_xyz[i], (224 * 2, 224 * 2), subplot=True, ind=ind,
                         f=fig, z_lim=z_lim, subplot_size=(row, col))
        ind += 1

        ax = fig.add_subplot(row, col, ind)
        ax.imshow(images[i])
        ax.axis('off')
        ind += 1

    plt.savefig(save_name)
    if visualize_3d:
        plt.show()
    plt.close(fig)


def plot_loss(loss, val_loss, path, title):
    plt.figure()
    x = np.arange(0, len(loss)) + 1
    plt.plot(x, loss, label='loss')
    plt.plot(x, val_loss, '--', label='validation loss')
    plt.legend()
    if title == 'Total loss':
        plt.ylim((0.1, 0.55))
    plt.title(title)
    plt.savefig(path)
    plt.close()


# plt.show()

def plot_result(save3D, save2D, freihand_path, result_path, version, visualize_3d=False):
    test = ""
    dataset = 'validation'
    if dataset == 'test':
        test = '_test'
    xyz_list, K_list, num_samples, s_list = get_raw_data(freihand_path, data_set=dataset)
    target_uv = []
    for i in range(num_samples):
        target_uv.append(projectPoints(xyz_list[i], K_list[i]))

    nbr_of_images = 5
    images = get_evalImages(freihand_path, nbr_of_images, dataset=dataset)

    if version == 'hm':
        hm_pred_file = result_path + 'hm_pred'+test+'.csv'
        hm_small_pred_file = result_path + 'hm_small_pred'+test+'.csv'
        try: #because my file is so large..
            hm = np.loadtxt(hm_small_pred_file, delimiter=',')
        except:
            hm = np.loadtxt(hm_pred_file, delimiter=',')
            #TODO: Save new uv for my hm predictions..
            np.savetxt(hm_small_pred_file, hm[0:21 * 10], delimiter=',')
        pred_hm = np.reshape(hm, (-1, 56, 56, 21))

    uv_pred_file = result_path + 'uv_pred'+test+'.csv'
    pred_uv = np.reshape(np.loadtxt(uv_pred_file, delimiter=','), (-1, 21, 2))
    print(pred_uv.shape)
    rec = True
    try:
        uv_pred_proj_file = result_path + 'reconstructed'+test+'/uv_pred_abs.csv'
        pred_uv_abs = np.reshape(np.loadtxt(uv_pred_proj_file, delimiter=','), (-1, 21, 2))
    except:
        print("reconstruction does not exist")
        rec = False
    ## plot heatmap for thumb
    try:
        if save2D and version=='hm':
            for i in range(nbr_of_images-5,nbr_of_images):
                plot_hm_with_images(pred_hm[i], target_uv[i], pred_uv[i], images[i], result_path,i)
    except:
        print('ERROR: to few heatmaps saved')
    # Plots first five
    if save2D:
        plot_skeleton(pred_uv[0:nbr_of_images], target_uv[0:nbr_of_images], images,result_path + 'images'+test+'/skeleton.png')
        if rec:
            plot_skeleton_all(pred_uv[0:nbr_of_images],pred_uv_abs[0:nbr_of_images], target_uv[0:nbr_of_images], images, result_path+ 'images'+test+'/skeleton_reprojected.png')

    # TODO: plot skeleton 3D - update with real file name
    #xyz_pred_file = result_path + 'pose_cam_xyz_pred_.csv'
    #xyz_tar_file = result_path + 'pose_cam_xyz_target_.csv'
    xyz_pred_file = result_path + 'xyz_pred'+test+'.csv'
    xyz_pred_abs_file = result_path + 'reconstructed'+test+'/xyz_pred_abs.csv'
    exist_3d = False
    target_xyz = np.array(xyz_list[:nbr_of_images])
    target_xyz_rel = np.copy(target_xyz)
    for i in range(len(target_xyz_rel)-5,len(target_xyz_rel)):
        target_xyz_rel[i][:,2] = target_xyz_rel[i][:,2]-target_xyz_rel[i][0,2]
    try:
        pred_xyz = np.reshape(np.loadtxt(xyz_pred_file, delimiter=','), (-1, 21, 3))
        #target_xyz_rel = np.reshape(np.loadtxt(xyz_tar_file, delimiter=','), (-1, 21, 3))
        exist_3d = True
        if rec:
            pred_xyz_abs = np.reshape(np.loadtxt(xyz_pred_abs_file, delimiter=','), (-1, 21, 3))

        #for i in range(len(target_xyz)):
        #   target_xyz_rel[i] = target_xyz[i]*s_list[i]

    except:
        print('No 3d results found')
   # nbr_of_images = 5
    # TODO: clean
    if exist_3d:
        try:
            loss_file = result_path + 'loss.csv'
            val_loss_file = result_path + 'val_loss.csv'
            loss = np.loadtxt(loss_file, delimiter=',')
            val_loss = np.loadtxt(val_loss_file, delimiter=',')
            title = 'Total loss'
            plot_loss(loss, val_loss, result_path + 'loss.png', title)
            loss_file = result_path + 'xyz_loss.csv'
            val_loss_file = result_path + 'val_xyz_loss.csv'
            loss = np.loadtxt(loss_file, delimiter=',')
            val_loss = np.loadtxt(val_loss_file, delimiter=',')
            title = '3D loss'
            plot_loss(loss, val_loss, result_path + 'loss_xyz.png', title)
            if version == 'hm':
                loss_file = result_path + 'hm_loss.csv'
                val_loss_file = result_path + 'val_hm_loss.csv'
                loss = np.loadtxt(loss_file, delimiter=',')
                val_loss = np.loadtxt(val_loss_file, delimiter=',')
                title = 'Heatmap loss'
                plot_loss(loss, val_loss, result_path + 'loss_hm.png', title)
            else:
                loss_file = result_path + 'loss_uv.csv'
                val_loss_file = result_path + 'loss_val_uv.csv'
                loss = np.loadtxt(loss_file, delimiter=',')
                val_loss = np.loadtxt(val_loss_file, delimiter=',')
                title = 'UV loss'
                plot_loss(loss, val_loss, result_path + 'loss_uv.png', title)

        except:
            print('No xyz or hm/uv loss file exist')

        if save3D:
            plot_3d_multiple(pred_xyz[nbr_of_images-5:nbr_of_images], target_xyz_rel[nbr_of_images-5:nbr_of_images], result_path, s_list[nbr_of_images-5:nbr_of_images], images[nbr_of_images-5:nbr_of_images], visualize_3d=visualize_3d, test=test)
            plot_3d(pred_xyz[nbr_of_images-5:nbr_of_images], target_xyz_rel[nbr_of_images-5:nbr_of_images], result_path, s_list[nbr_of_images-5:nbr_of_images], visualize_3d=visualize_3d, test=test)
            if rec:
                plot_3d_multiple(pred_xyz_abs[nbr_of_images-5:nbr_of_images], target_xyz[nbr_of_images-5:nbr_of_images], result_path,
                                 np.ones(nbr_of_images), images[nbr_of_images-5:nbr_of_images], visualize_3d=visualize_3d, reconstructed=True, test=test)
                plot_3d(pred_xyz_abs[nbr_of_images-5:nbr_of_images], target_xyz_rel[nbr_of_images-5:nbr_of_images], result_path,
                        np.ones(nbr_of_images), visualize_3d=visualize_3d, reconstructed=True, test=test)
    else:
        loss_file = result_path + 'loss.csv'
        val_loss_file = result_path + 'val_loss.csv'
        loss = np.loadtxt(loss_file, delimiter=',')
        val_loss = np.loadtxt(val_loss_file, delimiter=',')
        title = 'Total loss (2D)'
        plot_loss(loss, val_loss, result_path + 'loss.png', title)

if __name__ == '__main__':
    ''' Name of csv files that should exist in path: 
        xyz_pred.csv - 3D predictions with shape (-1, 21*3)
        uv_pred.csv - uv predictions with shape (-1, 21*2)
        hm_pred.csv - hm predictions with shape (-1,56*56)
        loss.csv - total loss
        val_loss.csv - total validation loss
        xyz_loss.csv - 3D loss
        xyz_val_loss.csv - 3D validation loss
        if version == 'uv'
            uv_loss.csv - 2D loss
            uv_val_loss.csv - 2D validation loss
        if version == 'hm'
            hm_loss.csv - heatmap loss
            hm_val_loss.csv - heatmap validation loss
    '''

    save3D = False
    save2D = True
    visualize_3d = False

    freihand_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/ED_50e_separated/'
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/ED_50e_shuffled/'
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/ED_50e_112/'
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/MN_50e/'
    #result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/hm/ED_30e_56/'
    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:", ["path="])
    except getopt.GetoptError:
        print('Require inputs -p <path> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-p':
            result_path = str(arg)
    version = 'uv'
    dir = 'images/'
    try:
        os.mkdir(os.path.join(result_path, dir))
    except OSError as error:
        print(error)
    plot_result(save3D, save2D, freihand_path, result_path,version, visualize_3d=visualize_3d)
    # TODO: Plot and project absolute result
