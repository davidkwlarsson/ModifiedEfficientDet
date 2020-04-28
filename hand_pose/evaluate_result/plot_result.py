import matplotlib.pyplot as plt
import numpy as np
from data_generators import get_raw_data

from help_functions import get_evalImages,projectPoints



def plot_hm_with_images(hm_pred, uv_true, uv_pred, img, path, ind):
    #fig = plt.figure()
    columns = 5
    rows = 2
    n = 0
    fig, axs = plt.subplots(rows,columns, figsize=(10, 3))

    for i in range(1,columns*2,2):
        axs[0,n].imshow(hm_pred[:, :,n])
       # axs[i, 0].colorbar()
        axs[0,n].scatter(uv_true[n,0]/4, uv_true[n,1]/4, c='r', s=2,label='true')
        axs[0,n].scatter(uv_pred[n,0], uv_pred[n,1], c='b', s=2,label='predicted')
        axs[1,n].imshow(img)
        #for i in range(21):
       # ind_pred = np.unravel_index(np.argmax(hm_pred[ind][:, :, n], axis=None), hm_pred[ind][:, :, n].shape)
       # ind_true = np.unravel_index(np.argmax(hm_true[ind][:, :, n], axis=None), hm_true[ind][:, :, n].shape)
        axs[1,n].scatter(uv_true[n,0], uv_true[n,1], c='r', s=2, label='true')
        axs[1,n].scatter(4 * uv_pred[n,0], 4 * uv_pred[n,1], c='b', s=2, label='predicted')
      #  print(ind_pred[1], ind_pred[0])
        n += 1
    plt.legend()
    plt.title('Thumb')
    #plt.show()
    plt.savefig(path+'images/hm_'+str(ind)+'.png')
    plt.close(fig)


def plot_loss(loss, val_loss, path):
   # plt.figure()
    x = np.arange(0,len(loss))+1
    plt.plot(x,loss, label='loss')
    plt.plot(x,val_loss, '--', label='validation loss')
    plt.legend()
   # plt.ylim((0.9, 0.11))
    plt.title('Heatmap loss, 2D')
    plt.savefig(path+'loss.png')
    plt.close()

# plt.show()


if __name__ == '__main__':
    freihand_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"
    xyz_list, K_list, num_samples, s_list = get_raw_data(freihand_path, data_set='validation')
    target_uv = []
    for i in range(num_samples):
        target_uv.append(projectPoints(xyz_list[i], K_list[i]))

    nbr_of_images = 1
    images = get_evalImages(freihand_path, nbr_of_images, dataset='validation')

    result_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/hm/ED_30e/'
    loss_file = result_path+'loss.csv'
    val_loss_file = result_path+'val_loss.csv'
    loss = np.loadtxt(loss_file, delimiter=',')
    val_loss = np.loadtxt(val_loss_file, delimiter=',')
    plot_loss(loss, val_loss, result_path)
    hm_pred_file = result_path+'hm_pred.csv'
    hm_small_pred_file = result_path+'hm_small_pred.csv'
    uv_pred_file = result_path+'uv_pred.csv'
    try:
        hm = np.loadtxt(hm_small_pred_file, delimiter=',')
    except:
        hm = np.loadtxt(hm_pred_file, delimiter=',')
        np.savetxt(hm_small_pred_file, hm[0:21*10], delimiter=',')

    pred_hm = np.reshape(hm, (-1, 56, 56, 21))
    pred_uv = np.reshape(np.loadtxt(uv_pred_file, delimiter=','), (-1, 21, 2))

    ## plot heatmap for thump
    for i in range(nbr_of_images):
        print(i)
        plot_hm_with_images(pred_hm[i], target_uv[i], pred_uv[i], images[i], result_path,i)

    # TODO: plot skeleton 2D
    # TODO: plot skeleton 3D