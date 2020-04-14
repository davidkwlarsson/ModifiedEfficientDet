import sys 
import os
import numpy as np
from numpy import loadtxt
from FreiHAND.freihand_utils import projectPoints

class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred, skip_check=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if not skip_check:
            keypoint_gt = np.squeeze(keypoint_gt)
            keypoint_pred = np.squeeze(keypoint_pred)
            keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

            assert len(keypoint_gt.shape) == 2
            assert len(keypoint_pred.shape) == 2
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff),axis = 1))

        num_kp = keypoint_gt.shape[1]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds


from help_functions import json_load


def get_uvlist(xyz, K):
    uv_list = []
    for i in range(10):
        uv = projectPoints(xyz[i], K[i])
        xy = []
        for coord in uv:
            xy.append([int(coord[0]), int(coord[1])])
        uv_list.append(np.array(xy))

    return uv_list
## CODE FOR EVALUATING THE OUTPUT TAKEN FROM FREIHAND :
#https://github.com/lmb-freiburg/freihand/blob/0808a4d29107ebcca2724b135728c1aa72022b77/eval.py#L199
def main():
    # Path to the target xyz coordinates
    dir_path = sys.argv[1]
    dim = sys.argv[2]        

    if dim == '2':
        xy_pred = loadtxt('pose_cam_xy.csv', delimiter=',')
        xy_pred = np.array(xy_pred).reshape(10,21,2)
        xyz = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:]
        K_list = K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
        uv_list = get_uvlist(xyz, K_list)
        xy = np.array(uv_list)
        xy = xy.reshape(10,21,2)
        print(np.sum(xy-xy_pred,axis = 0))
        eval_xy, eval_xy_aligned = EvalUtil(), EvalUtil()
        eval_xy.feed(
                xy,
                np.ones(xy.shape[1]),
                xy_pred
            )
        
        xy_mean2d, _, xy_auc2d, pck_xy, thresh_xy = eval_xy.get_measures(0.0, 0.05, 100)
        print('Evaluation 2D KP results:')
        print('auc=%.3f, mean_kp2d_avg=%.2f cm' % (xy_auc2d, xy_mean2d * 100.0))

    else:
        print("PREDICTIONS SHOULD BE SAVED AS : pose_cam_xyz.csv")
        xyz_pred = loadtxt('pose_cam_xyz.csv', delimiter=',') # Predictions
        xyz_pred = np.array(xyz_pred).reshape(-1,21,3)
        xyz = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:] # Target for validation data
            # Make sure they are equal in length
        xyz = np.array(xyz[:len(xyz_pred)]).reshape(-1,21,3)
        print(np.array(xyz_pred).shape, np.array(xyz).shape)
        eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
        for i in range(len(xyz_pred)):
            xyz_i = np.array(xyz[i])
            xyz_pred_i = np.array(xyz_pred[i])
            ## print(xyz_i.shape)
            # print(len(xyz_i.shape))
            eval_xyz.feed(
                xyz_i,
                np.ones(xyz_i.shape[0]),
                xyz_pred_i
            )
        # print(np.sum(xyz - xyz_pred, axis=0))
        print(np.sum(xyz-xyz_pred,axis = 0))
        xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D KP results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

import tensorflow as tf
from numpy import savetxt
from network import efficientdet
from help_functions import save_coords, add_depth_to_coords, add_relative
from FreiHAND.train_freihand_tfdata import tf_generator, get_session
from tensorflow.keras.optimizers import Adam

def add_focal(xyz_list, K_list):
    # xyz_list = np.reshape(xyz_list, (-1,21,3))
    xyz_new = []
    for i in range(len(xyz_list)):
        xy = np.array(xyz_list[i])[:, 0:-1]
        z = np.array(xyz_list[i])[:, 2]
        K = np.array(K_list[i])
        f = np.mean((K[0][0], K[1][1]))
        z = z * f #/ 100
        xyz = np.concatenate((xy, np.expand_dims(z, axis = -1)), axis = -1)
        print("check shape : ", np.shape(xyz))
        xyz = np.ndarray.flatten(xyz)
        xyz_new.append(xyz)
    return np.array(xyz_new)

def eval_with_model():
    dir_path = sys.argv[1]
    phi = 0
    weighted_bifpn = True
    freeze_backbone = True
    input_shape = (224,224,3)
    tf.compat.v1.keras.backend.set_session(get_session())
    print(tf.__version__)
    # images, heatmaps, heatmaps2,heatmaps3, coord = get_trainData(dir_path, 100, multi_dim=True)
    batch_size = 16
    nbr_epochs = 10
    num_samp = 128000
    num_val_samp = 1000
    train_dataset = tf_generator(dir_path, batch_size=batch_size, num_samp=num_samp, data_set = 'training')
    valid_dataset = tf_generator(dir_path, batch_size=batch_size, num_samp=num_val_samp, data_set = 'validation')
    traingen = train_dataset.prefetch(batch_size)
    validgen = valid_dataset.prefetch(batch_size)
    model = efficientdet(phi, input_shape = input_shape,
                        include_depth= False,
                        weighted_bifpn=weighted_bifpn,
                        freeze_bn=freeze_backbone)
    model.load_weights('model.h5')
    losses = {"uv_coords" : 'mean_squared_error', 'uv_depth' : 'mean_squared_error'}#, 'xyz_loss' : 'mean_squared_error'}
    alpha = 1.0 # tf.keras.backend.variable(1.0)
    beta = 1.0   # tf.keras.backend.variable(1.0)
    lossWeights = {"uv_coords" : alpha, 'uv_depth' : beta} #, 'xyz_loss' : 0.0}
    model.compile(optimizer = Adam(lr=1e-3),
                    loss = losses, loss_weights = lossWeights,
                    )
    
    preds, xyz_pred = model.predict(validgen, steps = 40)
    preds = (np.array(preds[:500]) + 1)*112
    print(np.shape(preds))
    preds = np.reshape(preds, (-1, 42))
    xyz_pred = xyz_pred[:500]
    s_list = json_load(os.path.join(dir_path, 'training_scale.json'))[-560:]
    K_list = json_load(os.path.join(dir_path, 'training_K.json'))[-560:]
    xyz_list = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:] # Target for validation data
            # Make sure they are equal in length
    xyz_list = np.array(xyz_list[:500])
    xyz_pred = add_relative(xyz_pred, xyz_list, s_list)
    # xyz_pred = add_depth_to_coords(preds, z_pred, K_list[:10], s_list[:10])
    print("shape of predictions : " , np.shape(xyz_pred))
    xyz_list = np.reshape(xyz_list, (-1, 63))
    print(np.sum(xyz_list - xyz_pred,axis = 0))
    # xyz_pred = add_focal(xyz_pred, K_list[:10])
    savetxt('pose_cam_xyz.csv',xyz_pred, delimiter=',')
    # print(xyz_pred[0])

if __name__ == '__main__':
    eval_with_model()
    main()