import getopt
import os
import sys

import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

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

def plot_hm_with_images(hm_pred, uv_true, uv_pred, img, path, ind, dir):
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
    plt.savefig(path + dir + 'hm_' + str(ind) + '.png')
    plt.close(fig)


def plot_skeleton_all(uv_preds,uv_preds_proj, images, path):
    plt.close()
    columns = 5
    rows = 3
    n = 0
    fig, axs = plt.subplots(rows, columns, figsize=(10, 8))
    print(uv_preds.shape)
    for i in range(1, columns * 2, 2):
        pred_skeleton = [4 * uv_preds[n][:, 0], 4 * uv_preds[n][:, 1]]
        proj_skeleton = [uv_preds_proj[n][:, 0], uv_preds_proj[n][:, 1]]
     #   true_skeleton = [uv_trues[n][:, 0], uv_trues[n][:, 1]]

        axs[0, n].imshow(images[n])
        axs[0, n].axis('off')

        plot_hand(axs[0, n], np.transpose(np.array(pred_skeleton)), order='uv')

        axs[1, n].imshow(images[n])
        axs[1, n].axis('off')
        plot_hand(axs[1, n], np.transpose(np.array(proj_skeleton)), order='uv')

        axs[2, n].imshow(images[n])
        axs[2, n].axis('off')
        #plot_hand(axs[2, n], np.transpose(np.array(true_skeleton)), order='uv')
        if n == 2:
            axs[0, n].set_title('Predictions')
            axs[1, n].set_title('Predictions, projected 3D')
            axs[2, n].set_title('True')

        n += 1
    fig.suptitle('Skeleton pose')

    plt.savefig(path)
   # plt.show()
    plt.close(fig)

def plot_skeleton(uv_preds, images, path):

    columns = 5
    rows = 2
    n = 0
    fig, axs = plt.subplots(rows, columns, figsize=(10, 3))
    for i in range(1, columns * 2, 2):
        pred_skeleton = [4 * uv_preds[n][:, 0], 4 * uv_preds[n][:, 1]]
     #   true_skeleton = [uv_trues[n][:, 0], uv_trues[n][:, 1]]

        axs[0, n].imshow(images[n])
        axs[0, n].axis('off')

        plot_hand(axs[0, n], np.transpose(np.array(pred_skeleton)), order='uv')

        axs[1, n].imshow(images[n])
        axs[1, n].axis('off')

       # plot_hand(axs[1, n], np.transpose(np.array(true_skeleton)), order='uv')
        if n == 2:
            axs[0, n].set_title('Predictions')
            axs[1, n].set_title('True')

        n += 1
    fig.suptitle('Skeleton pose')
    plt.savefig(path)
   # plt.show()
    plt.close(fig)


def plot_3d(pred_xyz, path,s,  visualize_3d=False, reconstructed=False, test='', dir='images/'):
    if reconstructed:
        fig, ax =plt.subplots()
      #  draw_3d_skeleton(target_xyz[2], (224 * 2, 224 * 2), f=fig)
        #plt.axis('off')
       # plt.minorticks_on()

        # Make a plot with major ticks that are multiples of 20 and minor ticks that
        # are multiples of 5.  Label major ticks with '%d' formatting but don't label
        # minor ticks.
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        plt.savefig('3d_skeleton_true.pdf', format='eps')
        plt.close()
    else:
        fig, ax = plt.subplots()
        draw_3d_skeleton(pred_xyz[2]*s[2], (224 * 2, 224 * 2), f=fig)
        plt.axis('off')
        # plt.minorticks_on()

        # Make a plot with major ticks that are multiples of 20 and minor ticks that
        # are multiples of 5.  Label major ticks with '%d' formatting but don't label
        # minor ticks.
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        plt.savefig('3d_skeleton_pred.pdf', format='eps')
        plt.close()
    if reconstructed:
        save_name = path + dir +'/3d_abs_'
    else:
        save_name = path + dir +'/3d_'

    for i in range(len(pred_xyz)):
        fig = plt.figure()
        fig, z_lim = draw_3d_skeleton(pred_xyz[i] * s[i], (224 * 2, 224 * 2), subplot=True, ind=1, f=fig)
        #plt.savefig(path + 'images/3d_pred_'+str(i)+'.png')
        draw_3d_skeleton(target_xyz[i], (224 * 2, 224 * 2), subplot=True, ind=2, f=fig, z_lim=z_lim)
        plt.savefig(save_name+str(i)+'.png')
        if visualize_3d:
            plt.show()

        plt.close(fig)


def plot_3d_multiple(pred_xyz, path,s, images, visualize_3d=False, reconstructed=False, test='', dir='images/'):
    fig = plt.figure(figsize=(8,10))
    row = len(pred_xyz)
    col = 3
    ind = 1
    if reconstructed:
        save_name = path + dir+'/3d_abs.png'
    else:
        save_name = path +dir+'/3d.png'

    for i in range(row):
        fig, z_lim = draw_3d_skeleton(pred_xyz[i] * s[i], (224 * 2, 224 * 2), subplot=True, ind=ind,
                                      f=fig, subplot_size=(row,col))
        ind += 1
       # draw_3d_skeleton(target_xyz[i], (224 * 2, 224 * 2), subplot=True, ind=ind,
                       #  f=fig, z_lim=z_lim, subplot_size=(row, col))
        ind += 1

        ax = fig.add_subplot(row, col, ind)
        ax.imshow(images[i])
        ax.axis('off')
        ind += 1

    plt.savefig(save_name)
    if visualize_3d:
        plt.show()
    plt.close(fig)


def plot_loss(loss, val_loss, path, title, acc=None, val_acc=None, dir='images/' ):
    plt.figure()
    x = np.arange(0, len(loss)) + 1
    plt.plot(x, loss, label='loss')
    plt.plot(x, val_loss, '--', label='validation loss')
    plt.legend()
    #if title == 'Total':
     #   plt.ylim((0, 0.55))
    plt.title(title + ' loss')
    plt.savefig(path+ dir +'loss_' + title+ '.pdf', format='pdf')
    plt.close()
    if acc is not None:
        plt.figure()
        x = np.arange(0, len(loss)) + 1
        plt.plot(x, acc, label='accuracy')
        plt.plot(x, val_acc, '--', label='validation accuracy')
        plt.legend()
        plt.title(title + ' accuracy')

        #if title == 'Total loss':
        #    plt.ylim((0.1, 0.55))
        plt.title(title)
        plt.savefig(path + dir + 'acc_' + title + '.pdf', format='pdf')
        plt.close()


# plt.show()

def plot_result(save3D, save2D, freihand_path, result_path, version, dataset, test, visualize_3d=False, dir='images/'):

    K_list, num_samples, s_list = get_raw_data(freihand_path, data_set=dataset)
   # for i in range(num_samples):
        #target_uv.append(projectPoints(xyz_list[i], K_list[i]))
        # Abs
        # Worst 11479, 129, 8223, 3385, best prediction:, 2700, 2648, 11335, 8079
        # Rel
        # Worst 6501, 11410, 3245, 9757, best prediction:, 1908, 2648, 2596, 3521
        # AL
        # Worst: 1049
        # 12266
        # 9902
        # 6646
        # best prediction:
        # 1908
        # 212
        # 5145
        # 5164
    if dir=='images_worst/':
        #6501
        #11410
        #3245
        #9757
        nbr_of_images = 13013+2
        ind1 = 12695
        #ind1 =129
        ind1 = 1049#al
        ind2 = 10243
        ind2 = 6501#abs
        #ind2=    1049
        ind3 = 9195#rel
       # ind3= 11479
        ind3 = 11479
        ind4 = 11410
        ind4 = 12266#al
        ind5 = 6501
        ind5 = 11410#abs
        # these are for MN 112
        ind1 = 2498 #abs # al
        ind2 =4898 #abs # al
        ind3 = 134 #abs  # al
        ind4 =7396 #abs   # al
        ind5 = 5302 #abs  # rel
        # these are for MN 224
        ind1 = 9757#al
        ind2 = 11025#al
        ind3 = 13013#al
        ind4 = 11159#al
        ind5 = 9763#rel

    elif dir == 'images_best/':
        #1908
        #2648
        #2596
        #3521
        ind1 =6768#al
        ind2 = 3093##al
        ind3 = 1463#al
        ind4 = 10024#al
        ind5 = 2648#abs
        ind1 = 5433  # al
        ind2 = 11483  ##al
        ind3 = 1000  # al
        ind4 = 1993  # al
        ind5 = 7143  # abs
        # these are for MN 224

        ind1 = 7666#al
        ind2 = 2274#al
        ind3 = 10922#al
        ind4 = 12129#al
        ind5 = 9839#al
        nbr_of_images = 11797+2


    ind1 = 0
    ind2 = 1
    ind3 = 2
    ind4 = 3
    ind5 = 4
    nbr_of_images=5

    nbr_of_images = np.max([ind1, ind2, ind3, ind4, ind5]) + 2

    images = get_evalImages(freihand_path, nbr_of_images, dataset=dataset)
    print(len(images))
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
        pred_uv_abs=[0,0,0,0,0,0,0,0]*10
    ## plot heatmap for thumb

    if save2D:
        pred_uv_show = pred_uv[nbr_of_images-5:nbr_of_images]
        images_show = images[nbr_of_images-5:nbr_of_images]
        pred_uv_show = np.array([pred_uv[ind1], pred_uv[ind2], pred_uv[ind3], pred_uv[ind4],pred_uv[ind5] ])
        pred_uv_abs_show = np.array([pred_uv_abs[ind1], pred_uv_abs[ind2], pred_uv_abs[ind3], pred_uv_abs[ind4],pred_uv_abs[ind5] ])
        images_show = np.array([images[ind1], images[ind2], images[ind3], images[ind4],images[ind5] ])
        plot_skeleton(pred_uv_show, images_show,result_path + dir+'/skeleton.png')
        if rec:
            plot_skeleton_all(pred_uv_show,pred_uv_abs_show, images_show, result_path+ dir+'/skeleton_reprojected.png')

    # TODO: plot skeleton 3D - update with real file name
    #xyz_pred_file = result_path + 'pose_cam_xyz_pred_.csv'
    #xyz_tar_file = result_path + 'pose_cam_xyz_target_.csv'
    xyz_pred_file = result_path + 'xyz_pred'+test+'.csv'
    xyz_pred_abs_file = result_path + 'reconstructed'+test+'/xyz_pred_abs.csv'
    exist_3d = False
#    target_xyz = np.array(xyz_list[:nbr_of_images])
   # target_xyz_rel = np.copy(target_xyz)
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
            title = 'Total'
            plot_loss(loss, val_loss, result_path , title)
            loss_file = result_path + 'xyz_loss.csv'
            val_loss_file = result_path + 'val_xyz_loss.csv'
            loss = np.loadtxt(loss_file, delimiter=',')
            val_loss = np.loadtxt(val_loss_file, delimiter=',')
            title = '3D'
            plot_loss(loss, val_loss, result_path, title)


            if version == 'hm':
                loss_file = result_path + 'hm_loss.csv'
                loss = np.loadtxt(loss_file, delimiter=',')
                val_loss = np.loadtxt(val_loss_file, delimiter=',')
                title = 'Heatmap'
                plot_loss(loss, val_loss, result_path, title)
                acc_file = result_path + 'hm_accuracy.csv'
                val_loss_file = result_path + 'val_hm_loss.csv'
                val_acc_file = result_path + 'val_hm_accuracy.csv'
                acc = np.loadtxt(acc_file, delimiter=',')
                val_acc = np.loadtxt(val_acc_file, delimiter=',')
                title = 'Heatmap'
                plot_loss(loss, val_loss, result_path, title, acc=acc, val_acc=val_acc)
            else:
                loss_file = result_path + 'loss_uv.csv'
                val_loss_file = result_path + 'loss_val_uv.csv'
                loss = np.loadtxt(loss_file, delimiter=',')
                val_loss = np.loadtxt(val_loss_file, delimiter=',')
                title = 'UV'
                plot_loss(loss, val_loss, result_path, title)
            acc_file = result_path + 'xyz_accuracy.csv'
            val_acc_file = result_path + 'val_xyz_accuracy.csv'
            acc = np.loadtxt(acc_file, delimiter=',')
            val_acc = np.loadtxt(val_acc_file, delimiter=',')
            plot_loss(loss, val_loss, result_path, title, acc=acc, val_acc=val_acc)

        except IOError as e:
            print(e)

        if save3D:
            pred_xyz_show = pred_xyz[nbr_of_images - 5:nbr_of_images]
#            target_xyz_rel_show = target_xyz_rel[nbr_of_images-5:nbr_of_images]
            s_show = s_list[nbr_of_images - 5:nbr_of_images]
            pred_xyz_show=np.array([pred_xyz[ind1], pred_xyz[ind2], pred_xyz[ind3], pred_xyz[ind4],pred_xyz[ind5]])
            s_show =np.array([s_list[ind1], s_list[ind2], s_list[ind3], s_list[ind4],s_list[ind5]])
            im_show = images[nbr_of_images-5:nbr_of_images]
            im_show=np.array([images[ind1], images[ind2], images[ind3], images[ind4],images[ind5] ])
           # target_xyz_rel_show= np.array([target_xyz_rel[ind1], target_xyz_rel[ind2], target_xyz_rel[ind3], target_xyz_rel[ind4],target_xyz_rel[ind5] ])
           # target_xyz_show = target_xyz[nbr_of_images - 5:nbr_of_images]
          #  target_xyz_show= np.array([target_xyz[ind1], target_xyz[ind2], target_xyz[ind3], target_xyz[ind4],target_xyz[ind5] ])
            pred_xyz_abs_show = pred_xyz_abs[nbr_of_images - 5:nbr_of_images]
            pred_xyz_abs_show = np.array([pred_xyz_abs[ind1], pred_xyz_abs[ind2], pred_xyz_abs[ind3], pred_xyz_abs[ind4],pred_xyz_abs[ind5] ])
            plot_3d_multiple(pred_xyz_show,
                             result_path, s_show,
                             im_show,
                             visualize_3d=visualize_3d, test=test, dir=dir)
           # plot_3d(pred_xyz_show, target_xyz_rel_show, result_path, s_show, visualize_3d=visualize_3d, test=test,  dir=dir)
            if rec:
                plot_3d_multiple(pred_xyz_abs_show, result_path,
                                 np.ones(nbr_of_images), images_show, visualize_3d=visualize_3d, reconstructed=True, test=test , dir=dir)
                #plot_3d(pred_xyz_abs_show, target_xyz_show, result_path,
                #        np.ones(nbr_of_images), visualize_3d=visualize_3d, reconstructed=True, test=test, dir=dir)
    else:
        loss_file = result_path + 'loss.csv'
        val_loss_file = result_path + 'val_loss.csv'
        loss = np.loadtxt(loss_file, delimiter=',')
        val_loss = np.loadtxt(val_loss_file, delimiter=',')
        title = '2D Total'
        plot_loss(loss, val_loss, result_path, title)

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

    save3D = True
    save2D = True
    visualize_3d = False

    freihand_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/ED_50e_separated/'
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/ED_50e_shuffled/'
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/ED_50e_224/'
    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/tmp/'
    #result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/hm/MN_30e/'
    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:", ["path="])
    except getopt.GetoptError:
        print('Require inputs -p <path> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-p':
            result_path = str(arg)
    version = 'hm'
    dir = 'images_worst/'

    dir = 'images/'
    dir = 'images_eval/'

    try:
        os.mkdir(os.path.join(result_path, result_path+dir))
    except OSError as error:
        print(error)
    try:
        os.mkdir(os.path.join(result_path, dir))
    except OSError as error:
        print(error)
    test = ""
    dataset = 'evaluation'
    if dataset == 'test':
        test = '_test'
        dir = 'images_test/'
    elif dataset == 'evaluation':
        test = '_eval'
        dir = 'images_eval/'


    # TODO: Plot and project absolute result


    plot_result(save3D, save2D, freihand_path, result_path,version,dataset, test, visualize_3d=visualize_3d, dir=dir)
