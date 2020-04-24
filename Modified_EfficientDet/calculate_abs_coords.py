import matplotlib
import numpy as np

from ModifiedEfficientDet.Modified_EfficientDet.data_generators import get_raw_data
from ModifiedEfficientDet.Modified_EfficientDet.generator import projectPoints
from ModifiedEfficientDet.Modified_EfficientDet.help_functions import get_evalImages
from ModifiedEfficientDet.Modified_EfficientDet.plot_3d_skeleton import draw_3d_skeleton
import matplotlib.pyplot as plt

from freihand.utils.fh_utils import plot_hand


def test(plot,xyz_abs, xyz, xyz_p, uv, uv_p, K, s,ind):
    '''Recreate 3d pose from uv and z coordinate. Uv must be really good if doing this.. '''
    #todo: divide this script in two
    uv_c = uv
   # print(uv[9])
   # print(uv[10])
    test_true_vals = False
   # test_false_vals = False
    if test_true_vals:
        # First check so can get correct result with true data
        #uv = projectPoints(np.array(xyz_abs), K)
        uv = uv

    else:
        # Then test with predicted
        uv = uv_p
   # print('compare')
  #  print(uv_p)
   # print(uv_p[9])
   # print(uv_p[10])

    xyz_abs = np.array(xyz_abs)/s
    xyz = np.array(xyz)
   # draw_3d_skeleton(xyz_abs,(224*2,224*2))
   # draw_3d_skeleton(xyz,(224*2,224*2))

    #uv_t = projectPoints(np.array(xyz), K)

  #  plt.figure()
  #  plt.imshow(images[0])
  #  plt.scatter(uv[:,0], uv[:,1], marker='o', s=2, label='predicted')
    #plt.scatter(uv_t[:,0], uv_t[:,1], marker='x', s=2,label='true')
  #  plt.legend()
    #plt.show()
    K[0][2] = K[0][2]#/4
    K[1][2] = K[1][2]#/4
    K[0][0] = K[0][0]#/4
    K[1][1] = K[1][1]#/4
    z_root = xyz_abs[0][2]
    z9 = xyz[9][2]+z_root
    z10 = xyz[10][2]+z_root
    x0 = K[0][2]
    y0 = K[1][2]
    f = K[0][0]

    x9 = (uv[9][0] - x0) / f
    y9 = (uv[9][1] - y0) / f
    x10 = (uv[10][0] - x0) / f
    y10 = (uv[10][1] - y0) / f
    #print(x9)
    #x9 = uv[9][0] #- x0) / f
    #y9 = uv[9][1] #- y0) / f
    #x10 = uv[10][0] #- x0) / f
    #y10 = uv[10][1] #- y0) / f
    if test_true_vals:
        X9 = xyz[9][0]
        X10 = xyz[10][0]
        Y9 = xyz[9][1]
        Y10 = xyz[10][1]
        Z9 = xyz[9][2]
        Z10 = xyz[10][2]
    else:
        X9 = xyz_p[9][0]
        X10 = xyz_p[10][0]
        Y9 = xyz_p[9][1]
        Y10 = xyz_p[10][1]
        Z9 = xyz_p[9][2]
        Z10 = xyz_p[10][2]

    zp9 = xyz_p[9][2] + z_root
    zp10 = xyz_p[10][2] + z_root
   # print(zp9,z9)
   # print('Distance between joint 9 and 10 should be 1:')
    if test_true_vals:
     #   print((x9 * z9 - x10 * z10) ** 2 + (y9 * z9 - y10 * z10) ** 2 + (z9 - z10) ** 2)  # true
        Zrn = xyz[9][2]#-z_root
        Zrm = xyz[10][2]#-z_root
    else:
       # print((x9 * zp9 - x10 * zp10) ** 2 + (y9 * zp9 - y10 * zp10) ** 2 + (zp9 - zp10) ** 2)

        Zrn = xyz_p[9][2]
        Zrm = xyz_p[10][2]
   # print((X9 - X10) ** 2 + (Y9 - Y10) ** 2 + (Z9 - Z10) ** 2)

    xn = x9
    xm = x10
    yn = y9
    ym = y10

    C = 1#(x9 * z9 - x10 * z10) ** 2 + (y9 * z9 - y10 * z10) ** 2 + (z9 - z10) ** 2
    a = np.square(xn - xm) + np.square(yn - ym)
    b = 2*((xn-xm)*(xn*Zrn-xm*Zrm)+(yn-ym)*(yn*Zrn-ym*Zrm))
   # b = Zrn * (np.square(xn) + np.square(yn) - xn * xm - yn * ym) + Zrm * (
   #             np.square(xm) + np.square(ym) - xn * xm - yn * ym)
    c = np.square(xn * Zrn - xm * Zrm) + np.square(yn * Zrn - ym * Zrm) + np.square(Zrn - Zrm) - np.square(C)
    Zroot = 0.5 * (-b + np.sqrt(np.square(b) - 4 * a * c)) / a
   # print(a, b, c)
    print('Zroot')
    print(Zroot*s)
    print('z_root')
    print(z_root*s)
    r = []
    if test_true_vals:
        Z = Zroot+xyz[:,2]
    else:
        Z = Zroot+xyz_p[:,2]
        xyz_p_new = np.copy(xyz_p)
       # xyz_p_new[:, 2] = r[ind]+xyz_p_new[:,2]
    #print(Z*uv[:,0])
    uv_tmp = np.array([Z*uv[:,0],Z*uv[:,1],Z*np.ones(21)])

    xyz_new = np.matmul(np.linalg.inv(np.array(K)), uv_tmp).T
    #print(xyz_new.shape)
    #xyz_new = xyz_p
   # xyz_new[:,2] = xyz_new[:,2] + z_root
   # if plot:
        #draw_3d_skeleton(xyz_p_new, (224 * 2, 224 * 2))
       # draw_3d_skeleton(xyz_new*s, (224 * 2, 224 * 2))
        #draw_3d_skeleton(xyz_abs, (224 * 2, 224 * 2))
    diff = []
    y = []
    std = []
    m = []
    for z in range(1,200,1):
       # m.append([])
       # std.append([])
        xyz_tmp = np.copy(xyz_p)

        xyz_tmp[:, 2] = z/5 + xyz_p[:, 2]
        uv_tmp = np.array(projectPoints(xyz_tmp, K))
        y.append(z/5)

        try:
            diff.append(np.sum(np.linalg.norm(uv_tmp - uv_p ,axis=1)))
            std.append(np.std(np.linalg.norm(uv_tmp - uv_p, axis=1)))
            m.append(np.mean(np.linalg.norm(uv_tmp - uv_p, axis=1)))
        #  print(uv_tmp-uv_p,)
        except:
            print('failed')
   # print(np.argmin(diff))
   # print('PRED depth',y[np.argmin(diff)])
    #print('REAL depth', z_root)
    #print('std')
   # print(std)
   # print(np.argmin(std))

    #print('m')
   # print(m)
   # print(np.argmin(m))

    r.append(y[np.argmin(diff)])
    if plot:
        #plt.figure()
        plt.plot(np.array(y)[10:], np.array(diff)[10:])
        plt.axvline(z_root, color='r')
        plt.axvline(y[np.argmin(diff)], color='b')
    #plt.show()

    xyz_calc_root = np.copy(xyz_p)
  #  print(r)
  #  print(ind)
    xyz_calc_root[:, 2] = r[0] + xyz_p[:, 2]
    xyz_true_root = np.copy(xyz_p)
    xyz_true_root[:, 2] = z_root + xyz_p[:, 2]
    print('Z pred', r[0] * s)
    print('Z true', z_root * s)
    uv_calc_root = np.array(projectPoints(xyz_calc_root, K))
    uv_true_root = np.array(projectPoints(xyz_true_root, K))
    plt.figure()
    pred_uv = [uv_p[:,0], uv_p[:,1]]

    pred_skeleton = [uv_calc_root[:,0], uv_calc_root[:,1]]
    correct_skeleton = [uv_c[:,0], uv_c[:,1]]
    if plot:
        try:
            images = get_evalImages(dir_path, 10, dataset='validation')
            plt.imshow(images[ind])
            plt.scatter(uv_true_root[:,0], uv_true_root[:,1], marker='x', s=10,label='true root val')
            plt.scatter(uv_p[:,0], uv_p[:,1], marker='+', s=10,label='predicted uv')
            plt.scatter(uv_c[:,0], uv_c[:,1], marker='o', s=10,label='true position')
            #plt.scatter(uv[:,0],uv[:,1], marker='o')
            plot_hand(plt, np.transpose(np.array(pred_uv)), order='uv')
           # plot_hand(plt, np.transpose(np.array(pred_skeleton)), order='uv')
            plt.legend()

            draw_3d_skeleton(xyz_calc_root*s, (224 * 2, 224 * 2))
            draw_3d_skeleton(xyz_abs*s, (224 * 2, 224 * 2))
            plt.show()

        except:
            print('failed')
    return uv_calc_root, xyz_calc_root, xyz_abs

def test_2(xyz_abs, xyz, xyz_p, uv, uv_p, K, s):
    '''Can we calculate root from result?'''
    xyz_abs = np.array(xyz_abs)/s
   # print(uv)
    K[0][2] = K[0][2]/4
    K[1][2] = K[1][2]/4
    K[0][0] = K[0][0]/4
    K[1][1] = K[1][1]/4
    uv = uv
    print(uv[0])
    uv_ = np.round(projectPoints(np.array(xyz_abs), K))
    uv_2 = np.array(projectPoints(np.array(xyz_abs), K))
    print(uv_[0])
    print(uv_2[0])
    #print(uv/4)
    z = []
    f = K[0][0]
    for i in range(21):
        X = xyz_abs[i][0]
        x = uv[i][0]
        Y = xyz_abs[i][1]
        y = uv[i][1]
        x0 = 112/4

        if int(x-x0) != 0 and int(y-x0) != 0 and np.abs(Y)>0.1 and np.abs(X)>0.1:
            Zx = f*X/(x-x0)#-xyz[i][2]
            Zy = f*Y/(y-x0)#-xyz[i][2]
           # print(xyz[i][2])
            # wierd value when i = 6 ???
            z.append(Zx)
            z.append(Zy)
    print(z)
    print('---')
   # print(np.median(z))
   # print(np.mean(z))
    print(xyz_abs[0][2])
    print('---')

    #matplotlib.use('TkAgg')
    #plt.hist(z, bins=20)
    #plt.show()


def test3(xyz_abs, xyz, xyz_p, uv, uv_p, K, s):
    '''Test which root is best one'''
    xyz_abs = np.array(xyz_abs)
    uv = projectPoints(np.array(xyz_abs), K)
    diff = np.matmul(K,xyz_abs.T)-np.array([uv[:,0], uv[:,1],xyz_abs[:,2]])
    print(diff)
    print(uv[0])
    print(np.matmul(K,xyz_abs.T)[0])

   # for z in range(0,1,0.05):


if __name__ == '__main__':
    dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"
    test_path=""
    test_path = "/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e/"
    test_path = "/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e_fewer_weights/"
    file = test_path+"pose_cam_xyz_pred_.csv"
    pred_xyz_coords = np.reshape(np.loadtxt(file, delimiter=','), (100,21,3))
    file = test_path+"pose_cam_xyz_target_.csv"
    target_xyz_coords = np.reshape(np.loadtxt(file, delimiter=','), (100,21,3))
    file = test_path+"uv_preds2.csv"
    pred_xy_coords = np.reshape(np.loadtxt(file, delimiter=','),(100,21,2))
    file = test_path+"uv_targets2.csv"
    target_xy_coords = np.reshape(np.loadtxt(file, delimiter=','),(100,21,2))
    width = 224
    height = 224
    #data_set = 'small_dataset'
    #input_shape = (width, height)
    xyz_list, K_list, num_samples, s_list = get_raw_data(dir_path, 'validation')
    uv_save = []
    xyz_save = []
    xyza_save = []
    plot = True
# test(xyz_list[1],target_xyz_coords[1],pred_xyz_coords[1], target_xy_coords[1]*4,pred_xy_coords[1]*4, K_list[1], s_list[1])
    for i in range(100):

        u, xyz, xyz_a = test(plot,xyz_list[i],target_xyz_coords[i],pred_xyz_coords[i], target_xy_coords[i]*4,pred_xy_coords[i]*4, K_list[i], s_list[i], i)
        uv_save.append(u)
        xyz_save.append(xyz)
        xyza_save.append(xyz_a)
    #  print(i)
    print(np.shape(np.array(xyz_save)))
    uv_save = np.reshape(np.array(uv_save), (100*21,2))
    xyz_save = np.reshape(np.array(xyz_save), (100*21,3))
    xyza_save = np.reshape(np.array(xyza_save), (100*21,3))
    np.savetxt('uv_preds_after.csv', np.array(uv_save), delimiter=',')
    np.savetxt('xyz_preds_after.csv', np.array(xyz_save), delimiter=',')
    np.savetxt('xyz_target_after.csv', np.array(xyza_save), delimiter=',')
