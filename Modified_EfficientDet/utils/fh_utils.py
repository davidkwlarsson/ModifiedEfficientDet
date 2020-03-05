from __future__ import print_function, unicode_literals
import numpy as np
import json
import os
import time
import skimage.io as io


""" Draw functions. """
def plot_hand(axis, coords_hw, vis=None, color_fixed=None, linewidth='1', order='hw', draw_kp=True):
    """ Plots a hand stick figure into a matplotlib figure. """
    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]

    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])

    colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :],  ms=2)



""" Dataset related functions. """
def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'


class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]


    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size*cls.valid_options().index(version)


