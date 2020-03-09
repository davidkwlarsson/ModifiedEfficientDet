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

            assert len(keypoint_gt.shape) == 3
            assert len(keypoint_pred.shape) == 3
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=0))

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
        xyz_pred = np.array(xyz_pred).reshape(10,21,3)
        xyz = json_load(os.path.join(dir_path, 'training_xyz.json'))[-560:] # Target for validation data
            # Make sure they are equal in length
        xyz = np.array(xyz[:10]).reshape(10,21,3)
        print(np.array(xyz_pred).shape, np.array(xyz).shape)
        eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
        eval_xyz.feed(
                xyz,
                np.ones(xyz.shape[1]),
                xyz_pred
            )
        print(np.sum(xyz-xyz_pred,axis = 0))
        xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D KP results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

if __name__ == '__main__':
    main()