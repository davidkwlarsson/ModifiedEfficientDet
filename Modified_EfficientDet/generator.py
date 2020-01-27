import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sys
import json
import cv2

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K): 
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def dump(projection_path, projection_list):
    """ Save projections into a json file. """
    # make sure its only lists
    for image_list in projection_list:
        for coord in image_list:
            print(coord)
            coord = coord.tolist()

    # save to a json
    with open(projection_path, 'w') as fo:
        json.dump([projection_list], fo)
    print('Dumped %d joints projections to %s' % (len(projection_list), projection_path))


def main():
    base_path = sys.argv[1]

    xyz_list = json_load(os.path.join(base_path, 'training_xyz.json'))
    K_list = json_load(os.path.join(base_path, 'training_K.json'))
    # scale_list = json_load(os.path.join(base_path, 'training_scale.json'))
    # imgs = list()

    print(len(xyz_list))

    for i in range(len(xyz_list[:1000])):
        uv = projectPoints(xyz_list[i], K_list[i])
        temp_im = np.zeros((224,224))
        # img = list()
        for j,coord in enumerate(uv):
            # temp_im = np.zeros((224,224))
            temp_im[int(coord[0]), int(coord[1])] = 255
            # img.append(temp_im)
        # imgs.append(temp_im)
        imagename = 'training/heatmaps/%08d.jpg' % i
        # print(imagename)
        cv2.imwrite(os.path.join(base_path, imagename),temp_im)

    # fig = plt.figure()
    # image = cv2.imread(os.path.join(base_path, 'training/rgb/00000000.jpg'))
    # extent = np.min(image), np.max(image), np.min(imgs[0]), np.max(imgs[0])
    # plt.imshow(image)
    # plt.imshow(imgs[0], alpha=.9)
    # plt.show()


    # dump(os.path.join(base_path, 'training_heatmaps.json'), imgs)

if __name__ == '__main__':
    main()