from __future__ import print_function, unicode_literals

import base64
import os

import matplotlib

from eval_util import EvalUtil

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pip
import argparse
import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
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


def createHTML(outputDir, file_name, curve_list):
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

    htmlfile = open(os.path.join(outputDir+file_name+ "_scores.html"), "w")
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

def evaluate_result(output_dir, save_path, xyz_pred, xyz_target, xy_pred, xy_target, got_3d=True):
    if got_3d:
        assert len(xyz_pred) == len(xyz_target), 'Expected format mismatch.'

        # init eval utils
        eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
        # eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)
        f_score, f_score_aligned = list(), list()
        f_threshs = [0.005, 0.015]

        shape_is_mano = False

        rng = len(xyz_pred)

        # iterate over the dataset once
        for idx in range(len(xyz_target)):
            xyz = np.array(xyz_target[idx])

            xyz_pred_i = np.array(xyz_pred[idx])

            # Not aligned errors

            eval_xyz.feed(
                xyz,
                np.ones(xyz.shape[0]),
                xyz_pred_i
            )

            # align predictions
            xyz_pred_aligned = align_w_scale(xyz, xyz_pred_i)
            # use trafo estimated from keypoints
            trafo = align_w_scale(xyz, xyz_pred_i, return_trafo=True)

            # Aligned errors
            eval_xyz_aligned.feed(
                xyz,
                np.ones(xyz.shape[0]),
                xyz_pred_aligned
            )
    assert len(xy_pred) == len(xy_target), 'Expected format mismatch.'

    eval_xy_aligned, eval_xy = EvalUtil(), EvalUtil()

    for idx in range(len(xy_target)):
        xy = np.array(xy_target[idx])
        # xy[:,0] = xy[:,0]+1
        # xyz = [np.array(x) for x in xyz]

        xy_p = np.array(xy_pred[idx])
        # print(xyz)
        # print(xyz_pred)

        # Not aligned errors
        eval_xy.feed(
            xy,
            np.ones(xy.shape[0]),
            xy_p
        )
        xy_aligned_p = align_w_scale(xy, xy_p)

        eval_xy_aligned.feed(
            xy,
            np.ones(xy.shape[0]),
            xy_aligned_p
        )
    if got_3d:
        # Calculate results
        xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 200)
        print('Evaluation 3D KP results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

        xyz_al_mean3d, _, xyz_al_auc3d, pck_xyz_al, thresh_xyz_al = eval_xyz_aligned.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D KP ALIGNED results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (xyz_al_auc3d, xyz_al_mean3d * 100.0))
    xy_mean2d, _, xy_auc2d, pck_xy, thresh_xy = eval_xy.get_measures(0.0, 20, 100)
    print('Evaluation 2D KP results:')
    # print('auc=%.3f, mean_kp2d_avg=%.2f px(?)' % (xy_auc2d, xy_mean2d * 100.0))
    print('auc=%.3f, mean_kp2d_avg=%.2f px(?)' % (xy_auc2d, xy_mean2d))

    xy_mean2d_al, _, xy_auc2d_al, pck_xy_al, thresh_xy_al = eval_xy_aligned.get_measures(0.0, 20, 100)
    print('Evaluation 2D KP ALIGNED results:')
    # print('auc=%.3f, mean_kp2d_avg=%.2f px(?)' % (xy_auc2d, xy_mean2d * 100.0))
    print('auc=%.3f, mean_kp2d_avg=%.2f px(?)' % (xy_auc2d_al, xy_mean2d_al))

   # print('F-scores')
   # f_out = list()
   # f_score, f_score_aligned = np.array(f_score).T, np.array(f_score_aligned).T
   # for f, fa, t in zip(f_score, f_score_aligned, f_threshs):
   #     print('F@%.1fmm = %.3f' % (t * 1000, f.mean()), '\tF_aligned@%.1fmm = %.3f' % (t * 1000, fa.mean()))
   #     f_out.append('f_score_%d: %f' % (round(t * 1000), f.mean()))
    #    f_out.append('f_al_score_%d: %f' % (round(t * 1000), fa.mean()))
#
    # Dump results
    score_path = os.path.join(output_dir,save_path)+'.txt'
    with open(score_path, 'w') as fo:
        if got_3d:

            xyz_mean3d *= 100
            xyz_al_mean3d *= 100
            fo.write('xyz_mean3d: %f\n' % xyz_mean3d)
            fo.write('xyz_auc3d: %f\n' % xyz_auc3d)
            fo.write('xyz_al_mean3d: %f\n' % xyz_al_mean3d)
            fo.write('xyz_al_auc3d: %f\n' % xyz_al_auc3d)
        fo.write('xy_mean2d: %f\n' % xy_mean2d)
        fo.write('xy_auc2d: %f\n' % xy_auc2d)
        fo.write('xy_al_mean2d: %f\n' % xy_mean2d_al)
        fo.write('xy_al_auc2d: %f\n' % xy_auc2d_al)
       # for t in f_out:
       #     fo.write('%s\n' % t)
    print('Scores written to: %s' % score_path)
    if got_3d:
        # scale to cm
        thresh_xyz *= 100.0
        thresh_xyz_al *= 100.0

        createHTML(
            output_dir, save_path,
            [
                curve(thresh_xyz, pck_xyz, 'Distance in cm', 'Percentage of correct keypoints',
                      'PCK curve for keypoint error'),
                curve(thresh_xyz_al, pck_xyz_al, 'Distance in cm', 'Percentage of correct keypoints',
                      'PCK curve for aligned keypoint error'),
                curve(thresh_xy, pck_xy, 'Distance in px', 'Percentage of correct keypoints',
                      'PCK curve for aligned keypoint error'),
                curve(thresh_xy_al, pck_xy_al, 'Distance in px', 'Percentage of correct keypoints',
                      'PCK curve for aligned keypoint error'),
            ]
        )
    else: #ugly solution..
        createHTML(
            output_dir, save_path,
            [
                curve(thresh_xy, pck_xy, 'Distance in px', 'Percentage of correct keypoints',
                      'PCK curve for aligned keypoint error'),
                curve(thresh_xy_al, pck_xy_al, 'Distance in px', 'Percentage of correct keypoints',
                      'PCK curve for aligned keypoint error'),
                curve(thresh_xy, pck_xy, 'Distance in px', 'Percentage of correct keypoints',
                      'PCK curve for aligned keypoint error'),
                curve(thresh_xy_al, pck_xy_al, 'Distance in px', 'Percentage of correct keypoints',
                      'PCK curve for aligned keypoint error'),
            ]
        )
    if got_3d:
        pck_curve_data = {
            'xyz': [thresh_xyz.tolist(), pck_xyz.tolist()],
            'xyz_al': [thresh_xyz_al.tolist(), pck_xyz_al.tolist()],
            'xy': [thresh_xy.tolist(), pck_xy.tolist()],
            'xy_al': [thresh_xy_al.tolist(), pck_xy_al.tolist()],
        }
    else:
        pck_curve_data = {
            'xy': [thresh_xy.tolist(), pck_xy.tolist()],
            'xy_al': [thresh_xy_al.tolist(), pck_xy_al.tolist()],
        }
    with open(output_dir + save_path + 'pck_data.json', 'w') as fo:
        json.dump(pck_curve_data, fo)

    print('Evaluation complete. Evaluated on %d samples' % len(xy_target))


def main(gt_path, pred_path, output_dir, pred_file_name=None, set_name=None):
    if pred_file_name is None:
        pred_file_name = 'pred.json'
    if set_name is None:
        set_name = 'evaluation'
    try:
        dir_path = sys.argv[2]
    except:
        print('no output_folder')
   # path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e_fewer_weights/'
    path = ""
    file_path =path+'pose_cam_xyz_pred_.csv'
    test_z_root = False
    if test_z_root:
        file_path ='xyz_preds_after.csv'


    dir_path2 = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"  #
    s_list = json_load(os.path.join(dir_path2, 'training_scale.json'))[-560:]
    pred = np.loadtxt(file_path, delimiter=',')
    xyz_list = np.loadtxt(path+'pose_cam_xyz_target_.csv', delimiter=',')
    xy_list = np.loadtxt(path+'uv_targets2.csv', delimiter=',').reshape(-1,21,2)*4
    xy_pred = np.loadtxt(path+'uv_preds2.csv', delimiter=',').reshape(-1, 21, 2)*4

    if test_z_root:
        xy_pred = np.loadtxt('uv_preds_after.csv', delimiter=',').reshape(-1,21,2)
        xyz_list = np.loadtxt('xyz_target_after.csv', delimiter=',')

   # print(xy_pred.shape)
   # # load eval annotations
    #xyz_list = json_load(os.path.join(gt_path, '%s_xyz.json' % set_name))
    n = int(len(pred)/21)
    pred = np.array(pred).reshape(-1, 21, 3)
    xyz_list = np.array(xyz_list).reshape(-1, 21,3)
    for i in range(xyz_list.shape[0]):
        pred[i] = pred[i] * s_list[i]
        xyz_list[i] = xyz_list[i] * s_list[i]
  #  print(len(pred))
  #  print(np.shape(pred))
    #assert len(pred) == 2, 'Expected format mismatch.'
    assert len(pred) == len(xyz_list), 'Expected format mismatch.'
    assert len(xy_pred) == len(xy_list), 'Expected format mismatch.'
    #assert len(pred[1]) == len(xyz_list), 'Expected format mismatch.'

    # init eval utils
    eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
   # eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)
    f_score, f_score_aligned = list(), list()
    f_threshs = [0.005, 0.015]

    shape_is_mano = False

    rng = len(pred)

    # iterate over the dataset once
    for idx in range(100):

        xyz = np.array(xyz_list[idx])
       # xyz = [np.array(x) for x in xyz]

        xyz_pred = np.array(pred[idx])
        #print(xyz)
        #print(xyz_pred)

       # xyz_pred = [np.array(x) for x in xyz_pred]
        #print('xyz_pred', np.shape(xyz_pred))
        #print('xyz', np.shape(xyz))
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
    eval_xy_aligned,eval_xy = EvalUtil(), EvalUtil()

    for idx in range(100):

        xy = np.array(xy_list[idx])
        #xy[:,0] = xy[:,0]+1
       # xyz = [np.array(x) for x in xyz]

        xy_p = np.array(xy_pred[idx])
        #print(xyz)
        #print(xyz_pred)


        # Not aligned errors
        eval_xy.feed(
            xy,
            np.ones(xy.shape[0]),
            xy_p
        )
        xy_aligned_p = align_w_scale(xy, xy_p)

        eval_xy_aligned.feed(
           xy,
           np.ones(xy.shape[0]),
           xy_aligned_p
       )



    # Calculate results
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 200)
    print('Evaluation 3D KP results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

    xyz_al_mean3d, _, xyz_al_auc3d, pck_xyz_al, thresh_xyz_al = eval_xyz_aligned.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP ALIGNED results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (xyz_al_auc3d, xyz_al_mean3d * 100.0))
    print('Wierd that this curve')
    xy_mean2d, _, xy_auc2d, pck_xy, thresh_xy = eval_xy.get_measures(0.0, 20, 100)
    print('Evaluation 2D KP results:')
    #print('auc=%.3f, mean_kp2d_avg=%.2f px(?)' % (xy_auc2d, xy_mean2d * 100.0))
    print('auc=%.3f, mean_kp2d_avg=%.2f px(?)' % (xy_auc2d, xy_mean2d))

    xy_mean2d_al, _, xy_auc2d_al, pck_xy_al, thresh_xy_al = eval_xy_aligned.get_measures(0.0, 20, 100)
    print('Evaluation 2D KP ALIGNED results:')
    # print('auc=%.3f, mean_kp2d_avg=%.2f px(?)' % (xy_auc2d, xy_mean2d * 100.0))
    print('auc=%.3f, mean_kp2d_avg=%.2f px(?)' % (xy_auc2d_al, xy_mean2d_al))

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
            curve(thresh_xy, pck_xy, 'Distance in px', 'Percentage of correct keypoints',
                  'PCK curve for aligned keypoint error'),
            curve(thresh_xy_al, pck_xy_al, 'Distance in px', 'Percentage of correct keypoints',
                  'PCK curve for aligned keypoint error'),
        ]
    )

    pck_curve_data = {
        'xyz': [thresh_xyz.tolist(), pck_xyz.tolist()],
        'xyz_al': [thresh_xyz_al.tolist(), pck_xyz_al.tolist()],
        'xy': [thresh_xy.tolist(), pck_xy.tolist()],
        'xy_al': [thresh_xy_al.tolist(), pck_xy_al.tolist()],
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
