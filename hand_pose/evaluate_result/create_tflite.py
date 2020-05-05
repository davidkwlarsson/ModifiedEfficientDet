import tensorflow as tf
import sys

sys.path.insert(1, '../')

from utils.losses import weighted_bce

from EfficientDet.layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization

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


#custom_objects = EfficientNetB0.custom_objects.copy()
#path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/50e_hm_feat_seperated/'
path = '/Users/Sofie/exjobb/ModifiedEfficientDet/whole_pipeline/im-xyz/mobilenet50e/'
path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/3d/MN_50e/'
path = '/Users/Sofie/exjobb/ModifiedEfficientDet/results/'

model = tf.keras.models.load_model(path+"saved_model_112/my_model", compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open(path+"model_ED_uv_112.tflite", "wb") .write(tfmodel)

