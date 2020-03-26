import os
import sys
import json

import tensorflow as tf
# from network import efficientdet
# from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
# from FreiHAND.freihand_utils import *
# from losses import *


from help_functions import load_model


tf.compat.v1.disable_eager_execution()
dir = sys.argv[1]

model = load_model(dir)
print(model.summary)
