import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import interpolate
from lift_pose import *

from ModifiedEfficientDet.Modified_EfficientDet.data_generators import get_raw_data, get_uv_data

if __name__ == '__main__':
    try:
        dir_path = sys.argv[1]
    except:
        dir_path = "/Users/Sofie/exjobb/freihand/FreiHAND_pub_v2/"  #
    data_set = 'small_dataset'
    xyz_list, K_list, num_samples = get_raw_data(dir_path, data_set)
    uv_list = get_uv_data(xyz_list, K_list, num_samples)
    uv_n_list = normalize(uv_list, (224,224))
    idx = 2
    uv = np.array(uv_n_list[idx]) # flip?
    uv_1 = np.array(uv_list[idx])
    xyz = np.array(xyz_list[idx])
    X = np.zeros((224, 224))
    Y = np.zeros((224, 224))
    valuesx = []
    valuesy = []
    valuesz = []
    xx = []
    yy = []
    for i in range(len(uv)):
        valuesx.append(xyz[i][0])
        valuesy.append(xyz[i][1])
        valuesz.append(xyz[i][2])
        x = uv[i][0]
        y = uv[i][1]
        xx.append(x)
        yy.append(y)
        x = int(np.round(uv_1[i][0]))
        y = int(np.round(uv_1[i][1]))
        X[x][y] = xyz[i][0]
        Y[x][y] = xyz[i][1]

    images = get_evalImages(dir_path, 10, dataset='training')

    matplotlib.use('TkAgg')
    points = uv
    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]
    grid_z1 = griddata(points, valuesx, (grid_x, grid_y), method='linear')
    grid_z2 = griddata(points, valuesy, (grid_x, grid_y), method='linear')
    grid_z3 = griddata(points, valuesz, (grid_x, grid_y), method='linear')
    print(grid_z1)
    plt.subplot(231)
    plt.imshow(grid_z1)#, extent=(-1, 1, -1, 1))
    plt.colorbar()

    plt.subplot(232)
    plt.imshow(X)
    plt.colorbar()
    plt.subplot(233)
    plt.imshow(images[idx])
    plt.subplot(234)
    plt.imshow(grid_z2)#, extent=(-1, 1, -1, 1))
    plt.colorbar()
    plt.subplot(235)
    plt.imshow(grid_z3)
    plt.colorbar()
    plt.subplot(236)
    draw_3d_skeleton(xyz, (224 * 2, 224 * 2))

    plt.show()
    # HERE
    ##Create regular grid of points
    #xi = np.arange(0, len(grid[0, :]), 1)
    #yi = np.arange(0, len(grid[:, 0]), 1)

    ## Grid irregular points to regular grid using delaunay triangulation

   # print(xx)
   # print('---')
   # print(yy)
   # print('---')
  #   print(values)
  #
  #   f = interpolate.interp2d(xx, yy, values, kind='linear')
  #   print(f(xx,yy))
  #   xnew = np.arange(-1, 1, 1e-1)
  #   ynew = np.arange(-1, 1, 1e-1)
  #   xx, yy = np.meshgrid(xnew, ynew)
  #   print('---')
  #
  #   z = f(xnew, ynew)
  #   print(z)
  # #  print('---')
  # #  print(xnew)
  # #  print('---')
  # #  print(ynew)
  #
  #   ax = plt.axes(projection='3d')
  #   ax.contour3D(xx, yy, z, 50, cmap='binary')
  #   ax.set_xlabel('x')
  #   ax.set_ylabel('y')
  #   ax.set_zlabel('z')
  #   plt.figure()
  #   plt.subplot(221)
  #   #plt.plot(xnew, z[0, :], 'b-')
  #   plt.imshow(z)
  #   plt.xlim((-1,1))
  #   plt.ylim((-1,1))
  #   plt.colorbar()
  #   #plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)
  #   plt.title('Original')
  #   plt.subplot(222)
  #   plt.imshow(X)
  #   plt.colorbar()
  #   fig = plt.figure()
  #
  #   #plt.imshow(z, origin='lower')
  #   plt.title('Interpolation')
  #   plt.show()
  #
