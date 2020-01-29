import os
import sys
import skimage.io as io
import tensorflow as tf
import numpy as np
import tensorflow as tf

from network import efficientdet
from tensorflow.compat.v1.keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_testData(dir_path, num_samples = 100):
    print("Collecting data ... \n")
    imgs = []
    heats = []
    n = 0
    imgs_path = os.path.join(dir_path, 'training/rgb')
    for f in os.listdir(imgs_path):
        imgs.append(io.imread(os.path.join(imgs_path, f)))
        n += 1
        if n > num_samples:
            n = 0
            break

    heat_path = os.path.join(dir_path, 'training/heatmaps')
    for f in os.listdir(heat_path):
        heats.append(io.imread(os.path.join(heat_path, f)))
        n += 1
        if n > num_samples:
            n = 0
            break

    return np.array(imgs), np.array(heats)

def save_preds(dir_path, predictions):
    #Function that saves away the predictions as jpg images.
    save_path = dir_path + "predictions/test"
    makedirs(save_path)
    print("Saving the predictions to " + save_path)
    for i,pred in enumerate(predictions):
        name = save_path + "%s" % i
        io.imsave(name, pred)


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def get_session():
    """
    Construct a modified tf session.
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def main():
    dir_path = sys.argv[1]
    model_path = sys.argv[2]
    phi = 0
    weighted_bifpn = True
    freeze_backbone = False
    #Load test data
    images, heatmaps = get_testData(dir_path)
    tf.compat.v1.keras.backend.set_session(get_session())

    pred_model = efficientdet(phi, weighted_bifpn=weighted_bifpn,
                            freeze_bn=freeze_backbone)
    print("Load model ........ \n")
    pred_model.load_weights(model_path, by_name = True)
    pred_model.compile(optimizer=Adam(lr=1e-3),
                    loss='mse')
    preds = pred_model.predict(images)
    save_preds(dir_path, preds)


if __name__ == '__main__':
    main()
