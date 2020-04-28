import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.fh_utils import *
import skimage.io as io
from utils.help_functions import *
import pickle


def evaluate_result(dir_path, preds):
    predictions = []
    n = 0

    # Sum all heatmaps for the joints of one hand and save as figure
    for sample in preds:
        predictions.append(sample)
        sum = 0
        for k in range(21):
            #  print(np.shape(sample))
            sum = sum + sample[:, :, k]
        plt.figure()
        plt.imshow(sum)
        plt.colorbar()
        plt.savefig('tf_sum_fig_preds_hm' + str(n) + '.png')
        n = n + 1

    # From the predictions extract coordinates
    coord_preds = heatmaps_to_coord(predictions)
    # Get the images that corresponds to test data
    # Get the true coordinates
    images, uv, hm, c = get_data(dir_path, 10)
    coord = uv
    #images = get_evalImages(dir_path, 10)
    # Plot the predicted result as skeleton
    plot_predicted_hands_uv(images, coord_preds, 'hands_as_uv_pred.png')
    # Plot the images together with true and predicted heatmap
    plot_hm_with_images(predictions, hm, images, 4)
    # Plot of hand
    plot_predicted_coordinates(images, coord_preds, coord)



def plot_heatmaps_with_coords(images, heatmaps, coords):
    """Here the heatmaps are displayed together with the coordinates to
    check that the heatmaps are correctly generated"""
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
  #print(coords[0][0:2])
    plt.show()


def save_loss(history):
    """Plot acc and loss vs epochs"""
    #acc = history.history['acc']
    #val_acc = history.history['val_acc']
    print(history.history.keys())
    for k in history.history.keys():
        values = history.history[k]
        np.savetxt(str(k)+'.csv', values, delimiter=',')

    loss = history.history['loss']
    #xyz_loss = history.history['xyz_loss']
    #val_xyz_loss = history.history['val_xyz_loss']
    val_loss = history.history['val_loss']
    #lr = history.history['lr']
    epochs = range(1, len(loss) + 1)
    #np.savetxt('loss.csv', loss, delimiter=',')
    #np.savetxt('xyz_loss.csv', xyz_loss, delimiter=',')
    #np.savetxt('val_loss.csv', val_loss, delimiter=',')
    #np.savetxt('val_xyz_loss.csv', val_xyz_loss, delimiter=',')
    #np.savetxt('epochs.csv', epochs, delimiter=',')
    #np.savetxt('lr.csv', lr, delimiter=',')

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
    plt.figure()
    plt.plot(epochs[1:], loss[1:], 'bo', label='Training loss')
    plt.plot(epochs[1:], val_loss[1:], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    plt.savefig('acc_loss_2.png')




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
    plt.figure()
    plt.imshow(np.sum(heatmaps[0][:, :, :], axis=-1))
    plt.savefig('hand_as_hm.png')


def plot_hm_with_images(hm_pred, hm_true, img, ind=0, scale=1):
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 5
    n = 0
    for i in range(1,rows*3,3):
        fig.add_subplot(rows, columns, i)
       # print(ind)
       # print(n)
        plt.imshow(hm_pred[ind][:, :, n])
        plt.colorbar()
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(hm_true[ind][:, :, n])
        ind_pred = np.unravel_index(np.argmax(hm_pred[ind][:, :, n], axis=None), hm_pred[ind][:, :, n].shape)
        ind_true = np.unravel_index(np.argmax(hm_true[ind][:, :, n], axis=None), hm_true[ind][:, :, n].shape)
        plt.scatter(ind_pred[1], ind_pred[0], c='r', s=2)
        plt.scatter(ind_true[1], ind_true[0], c='b', s=2)
       # print('-----')
       # print(ind_pred[1], ind_pred[0])

        plt.colorbar()
        fig.add_subplot(rows, columns, i + 2)
        plt.imshow(img[ind])
        #for i in range(21):
       # ind_pred = np.unravel_index(np.argmax(hm_pred[ind][:, :, n], axis=None), hm_pred[ind][:, :, n].shape)
       # ind_true = np.unravel_index(np.argmax(hm_true[ind][:, :, n], axis=None), hm_true[ind][:, :, n].shape)
        plt.scatter(scale * ind_pred[1], scale * ind_pred[0], c='r', s=2)
        plt.scatter(scale * ind_true[1], scale * ind_true[0], c='b', s=2)
      #  print(ind_pred[1], ind_pred[0])
        n += 1
    #plt.show()
    plt.savefig('extended_hm'+str(ind)+'.png')
    plt.close(fig)



def plot_predicted_heatmaps_tmp(preds, heatmaps):
  #  print(preds.shape)
  #  print(heatmaps.shape)
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
        plot_hand(ax, np.transpose(np.array(one_pred)))


    #plt.show()
    plt.savefig('hands.png')



def plot_predicted_hands_uv(images, coord_preds, name):
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        try:
            plt.imshow(images[i - 1])
            # need to project onto image..
            one_pred = [coord_preds[i-1][0::2], coord_preds[i-1][1::2]]
            plot_hand(ax, np.transpose(np.array(one_pred)),order='uv')
        except:
            print('error')
            continue
    plt.savefig(name)
  #plt.show()





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
   # matplotlib.use('TkAgg')

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
    ax.set_xlim(-0.08, 0.08)
    ax.set_ylim(-0.08, 0.08)
    ax.set_zlim(-0.08, 0.08)
    ax.view_init(elev=-90, azim=-90)
    # pickle.dump(fig, open('3DHands.fig.pickle', 'wb'))
    plt.savefig('hands_3D.png')
    plt.show()
    # ret = fig2data(fig)  # H x W x 4
    #plt.close(fig)
    # return ret
def save_z_root_coords(root_coord_pred, root_coord, name):
    np.savetxt('pose_cam_z_root_' + name + '.csv', root_coord, delimiter=',')
    np.savetxt('pose_cam_z_root_pred_' + name + '.csv', root_coord_pred, delimiter=',')


def save_coords_one_file(pose_cam_coord, name):
    #print('in save coords')
    pose_cam_coord = np.array(pose_cam_coord)
    print('pose_cam.shape', pose_cam_coord.shape)
    l, n, m = pose_cam_coord.shape
   # print(pose_cam_coord.shape)
   # l = 1
    pose_cam_coord = pose_cam_coord.reshape((l*n, m))
  #  print(pose_cam_coord.shape[-1])
    if pose_cam_coord.shape[-1] == 3:
       # print('in pose cam = 3')
        np.savetxt('pose_cam_xyz_'+name+'.csv',pose_cam_coord, delimiter=',')
       # pickle.dump(image, open('hand_for_3d.fig.pickle', 'wb'))

    elif pose_cam_coord.shape[-1] == 2:
        np.savetxt('pose_cam_xy_'+name+'.csv',pose_cam_coord, delimiter=',')
       # pickle.dump(image, open('hand_for_2d.fig.pickle', 'wb'))



def save_coord(pose_cam_coord, image, name):
    #print('in save coords')
    pose_cam_coord = np.array(pose_cam_coord)
    print('pose_cam.shape', pose_cam_coord.shape)
    n,m= pose_cam_coord.shape
   # print(pose_cam_coord.shape)
    l = 1
    pose_cam_coord = pose_cam_coord.reshape((l*n, m))
  #  print(pose_cam_coord.shape[-1])
    if pose_cam_coord.shape[-1] == 3:
       # print('in pose cam = 3')
        np.savetxt('pose_cam_xyz_'+name+'.csv',pose_cam_coord, delimiter=',')
        pickle.dump(image, open('hand_for_3d.fig.pickle', 'wb'))

    elif pose_cam_coord.shape[-1] == 2:
        np.savetxt('pose_cam_xy_'+name+'.csv',pose_cam_coord, delimiter=',')
        pickle.dump(image, open('hand_for_2d.fig.pickle', 'wb'))

   # coords = np.loadtxt('pose_cam_xyz.csv', delimiter=',')
   # print(coords)
    #draw_3d_skeleton(coords, (56,56))
   # print(image.shape)
   # print('draw_3d_skeleton done')