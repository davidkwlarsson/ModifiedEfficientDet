import tensorflow as tf
import efficientnet
from losses import weighted_bce
import keras.losses

from layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization
from keras.utils.generic_utils import get_custom_objects

custom_objects = {
#'BatchNormalization': BatchNormalization(freeze=False),
#'swish' : efficientnet.get_swish(backend=keras.backend,layers=keras.layers,models=keras.models,utils=keras.utils),
#'FixedDropout' : efficientnet.get_dropout(backend=keras.backend,layers=keras.layers,models=keras.models,utils=keras.utils),
'wBiFPNAdd' : wBiFPNAdd,
'weighted_bce' : weighted_bce
#'PriorProbability' : initializers.PriorProbability,
#'RegressBoxes' : layers_new.RegressBoxes,
#'FilterDetections' : layers_new.FilterDetections,
#'ClipBoxes' : layers_new.ClipBoxes,
#'_smooth_l1' : losses.smooth_l1(),
#'_focal' : losses.focal(),
}


from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6
#custom_objects = EfficientNetB0.custom_objects.copy()
#path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e_hm_feat_seperated/'
path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/mobilenet50e/'

model = tf.keras.models.load_model(path+"saved_model/my_model", compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open(path+"model_hm_mobilenet.tflite", "wb") .write(tfmodel)

