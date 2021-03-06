from __future__ import print_function, unicode_literals

import base64
import os
import sys

import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pip
import argparse
import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
from utils.fh_utils import *
from help_functions import json_load
from eval import EvalUtil
#from freihand.utils.eval_util import EvalUtil



def verts2pcd(verts, color=None):
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(verts)
    if color is not None:
        if color == 'r':
            pcd.paint_uniform_color([1, 0.0, 0])
        if color == 'g':
            pcd.paint_uniform_color([0, 1.0, 0])
        if color == 'b':
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)
    d1 = o3d.compute_point_cloud_to_point_cloud_distance(gt, pr)  # closest dist for each gt point
    d2 = o3d.compute_point_cloud_to_point_cloud_distance(pr, gt)  # closest dist for each pred point
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(
            len(d2))  # how many of our predicted points lie close to a gt point?
        precision = float(sum(d < th for d in d1)) / float(len(d1))  # how many of gt points are matched?

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall


def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def align_by_trafo(mtx, trafo):
    t2 = mtx.mean(0)
    mtx_t = mtx - t2
    R, s, s1, t1 = trafo
    return np.dot(mtx_t, R.T) * s * s1 + t1 + t2


class curve:
    def __init__(self, x_data, y_data, x_label, y_label, text):
        self.x_data = x_data
        self.y_data = y_data
        self.x_label = x_label
        self.y_label = y_label
        self.text = text


def createHTML(outputDir, curve_list):
    curve_data_list = list()
    for item in curve_list:
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(item.x_data, item.y_data)
        ax.set_xlabel(item.x_label)
        ax.set_ylabel(item.y_label)
        img_path = os.path.join(outputDir, "img_path_path.png")
        plt.savefig(img_path, bbox_inches=0, dpi=300)

        data_uri = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
        img_tag1 = '<img src="data:image/png;base64,{0}" alt="FROC" width="572pt" height="433pt">'.format(data_uri)
       # print(img_path1)
        # write image and create html embedding
       # data_uri1 = open(img_path, 'br').read()
       # data_uri1 = base64.b64encode(data_uri1).replace(b'\n', b'')

        #img_tag1 = 'src="data:image/png;base64,{0}"'.format(data_uri1)
        #print(img_tag1)
        curve_data_list.append((item.text, img_tag1))

        #os.remove(img_path)

    htmlString = '''<!DOCTYPE html>
    <html>
    <body>
    <h1>Detailed results:</h1>'''

    for i, (text, img_embed) in enumerate(curve_data_list):
        htmlString += '''
        <h2>%s</h2>
        <p>
        %s
        </p>
        <p>Raw curve data:</p>
        <p>x_axis: <small>%s</small></p>
        <p>y_axis: <small>%s</small></p>
        ''' % (text, img_embed, curve_list[i].x_data, curve_list[i].y_data)

    htmlString += '''
    </body>
    </html>'''

    htmlfile = open(os.path.join(outputDir, "scores.html"), "w")
    htmlfile.write(htmlString)
    htmlfile.close()


def _search_pred_file(pred_path, pred_file_name):
    """ Tries to select the prediction file. Useful, in case people deviate from the canonical prediction file name. """
    pred_file = os.path.join(pred_path, pred_file_name)
    if os.path.exists(pred_file):
        # if the given prediction file exists we are happy
        return pred_file

    print('Predition file "%s" was NOT found' % pred_file_name)

    # search for a file to use
    print('Trying to locate the prediction file automatically ...')
    files = [os.path.join(pred_path, x) for x in os.listdir(pred_path) if x.endswith('.json')]
    if len(files) == 1:
        pred_file_name = files[0]
        print('Found file "%s"' % pred_file_name)
        return pred_file_name
    else:
        print('Found %d candidate files for evaluation' % len(files))
        raise Exception('Giving up, because its not clear which file to evaluate.')


def main(gt_path, pred_path = None, output_dir = None, pred_file_name=None, set_name=None):
    if pred_file_name is None:
        pred_file_name = 'pred.json'
    if set_name is None:
        set_name = 'evaluation'
    # try:
    #     dir_path = sys.argv[2]
    # except:
    #     print('no output_folder')
    # file_path ='pose_cam_xyz_pred_.csv'
    dir_path = "pck_curve"
    output_dir = dir_path
    #file_path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e/' + file_path

    dir_path2 = "../../data/FreiHAND_pub_v2/"  #
    pred = np.loadtxt('pose_cam_xyz.csv', delimiter=',') # Predictions
    # xyz_pred = np.array(xyz_pred).reshape(-1,21,3)
    xyz_list = json_load(os.path.join(dir_path2, 'training_xyz.json'))[-560:]
    xyz_list = xyz_list[:500]
    # load eval annotations
    #xyz_list = json_load(os.path.join(gt_path, '%s_xyz.json' % set_name))
    # n = int(len(pred)/21)
    pred = np.array(pred).reshape(-1, 21, 3)
    xyz_list = np.array(xyz_list).reshape(-1, 21,3)
    # for i in range(xyz_list.shape[0]):
    #     pred[i] = pred[i] * s_list[i]
    #     xyz_list[i] = xyz_list[i] * s_list[i]
    print(len(pred))
    print(np.shape(pred))
    #assert len(pred) == 2, 'Expected format mismatch.'
    assert len(pred) == len(xyz_list), 'Expected format mismatch.'
    #assert len(pred[1]) == len(xyz_list), 'Expected format mismatch.'

    # init eval utils
    eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
   # eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)
    f_score, f_score_aligned = list(), list()
    f_threshs = [0.005, 0.015]

    shape_is_mano = False

    rng = len(pred)
    # iterate over the dataset once
    for idx in range(len(pred)):

        xyz = np.array(xyz_list[idx])
       # xyz = [np.array(x) for x in xyz]

        xyz_pred = np.array(pred[idx])
        # print(xyz)
        # print(xyz_pred)

       # xyz_pred = [np.array(x) for x in xyz_pred]
        # print('xyz_pred', np.shape(xyz_pred))
        # print('xyz', np.shape(xyz))
        # Not aligned errors

        eval_xyz.feed(
            xyz,
            np.ones(xyz.shape[0]),
            xyz_pred
        )

        # align predictions
        xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
           # use trafo estimated from keypoints
        trafo = align_w_scale(xyz, xyz_pred, return_trafo=True)

        # Aligned errors
        eval_xyz_aligned.feed(
            xyz,
            np.ones(xyz.shape[0]),
            xyz_pred_aligned
        )


    # Calculate results
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

    xyz_al_mean3d, _, xyz_al_auc3d, pck_xyz_al, thresh_xyz_al = eval_xyz_aligned.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP ALIGNED results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (xyz_al_auc3d, xyz_al_mean3d * 100.0))


    print('F-scores')
    f_out = list()
    f_score, f_score_aligned = np.array(f_score).T, np.array(f_score_aligned).T
    for f, fa, t in zip(f_score, f_score_aligned, f_threshs):
        print('F@%.1fmm = %.3f' % (t * 1000, f.mean()), '\tF_aligned@%.1fmm = %.3f' % (t * 1000, fa.mean()))
        f_out.append('f_score_%d: %f' % (round(t * 1000), f.mean()))
        f_out.append('f_al_score_%d: %f' % (round(t * 1000), fa.mean()))

    # Dump results
    score_path = os.path.join(output_dir, 'scores.txt')
    with open(score_path, 'w') as fo:
        xyz_mean3d *= 100
        xyz_al_mean3d *= 100
        fo.write('xyz_mean3d: %f\n' % xyz_mean3d)
        fo.write('xyz_auc3d: %f\n' % xyz_auc3d)
        fo.write('xyz_al_mean3d: %f\n' % xyz_al_mean3d)
        fo.write('xyz_al_auc3d: %f\n' % xyz_al_auc3d)

        for t in f_out:
            fo.write('%s\n' % t)
    print('Scores written to: %s' % score_path)

    # scale to cm
    thresh_xyz *= 100.0
    thresh_xyz_al *= 100.0

    createHTML(
        output_dir,
        [
            curve(thresh_xyz, pck_xyz, 'Distance in cm', 'Percentage of correct keypoints',
                  'PCK curve for keypoint error'),
            curve(thresh_xyz_al, pck_xyz_al, 'Distance in cm', 'Percentage of correct keypoints',
                  'PCK curve for aligned keypoint error'),
        ]
    )

    pck_curve_data = {
        'xyz': [thresh_xyz.tolist(), pck_xyz.tolist()],
        'xyz_al': [thresh_xyz_al.tolist(), pck_xyz_al.tolist()],
       }
    with open('pck_data.json', 'w') as fo:
        json.dump(pck_curve_data, fo)

    print('Evaluation complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('input_dir', type=str,
                        help='Path to where prediction the submited result and the ground truth is.')
    parser.add_argument('output_dir', type=str,
                        help='Path to where the eval result should be.')
    parser.add_argument('--pred_file_name', type=str, default='pred.json',
                        help='Name of the eval file.')
    args = parser.parse_args()

    # call eval
    main(
        os.path.join(args.input_dir, 'ref'),
        os.path.join(args.input_dir, 'res'),
        args.output_dir,
        args.pred_file_name,
        set_name='evaluation'
    )