import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

from model import *

os.environ["CUDA_VISIBLE_DEVICES"]=""

strategy = tf.device('/cpu:0')

weight = "/home/lap14880/hieunmt/faceid/nonmask_tfic/tf_image_classification/best_model_motor_reid_ConvNeXtTiny_160_458528.h5"
name = weight.split('/')[-1]
encoder_save = f"/home/lap14880/hieunmt/faceid/nonmask_tfic/tf_image_classification/convert/encoder/encoderv2_{name}"

im_size = 160
img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)

base_name = 'ConvNeXtTiny'

final_dropout = 0.1
have_emb_layer = True
emb_dim = 256
use_normdense = True

n_labels = 458528
cate_int = True

from keras_cv_attention_models import convnext

with strategy:
    base = get_base_model(base_name, input_shape)
    emb_model = create_emb_model(base, final_dropout, have_emb_layer, emb_dim)
    model = create_model(input_shape, emb_model, n_labels, use_normdense, cate_int)
    model.load_weights(weight)
    model.summary()

# class ChannelAffine(tf.keras.layers.Layer):
#     def __init__(self, use_bias=True, weight_init_value=1, axis=-1, **kwargs):
#         super(ChannelAffine, self).__init__(**kwargs)
#         self.use_bias, self.weight_init_value, self.axis = use_bias, weight_init_value, axis
#         self.ww_init = keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
#         self.bb_init = "zeros"
#         self.supports_masking = False

#     def build(self, input_shape):
#         if self.axis == -1 or self.axis == len(input_shape) - 1:
#             ww_shape = (input_shape[-1],)
#         else:
#             ww_shape = [1] * len(input_shape)
#             axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
#             for ii in axis:
#                 ww_shape[ii] = input_shape[ii]
#             ww_shape = ww_shape[1:]  # Exclude batch dimension

#         self.ww = self.add_weight(name="weight", shape=ww_shape, initializer=self.ww_init, trainable=True)
#         if self.use_bias:
#             self.bb = self.add_weight(name="bias", shape=ww_shape, initializer=self.bb_init, trainable=True)
#         super(ChannelAffine, self).build(input_shape)

#     def call(self, inputs, **kwargs):
#         return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def get_config(self):
#         config = super(ChannelAffine, self).get_config()
#         config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value, "axis": self.axis})
#         return config

# class NormDense(tf.keras.layers.Layer):
#     def __init__(self, units=1000, **kwargs):
#         super(NormDense, self).__init__(**kwargs)
#         self.init = keras.initializers.glorot_normal()
#         self.units = units

#     def build(self, input_shape):
#         self.w = self.add_weight(name="norm_dense_w", shape=(input_shape[-1], self.units), initializer=self.init, trainable=True)
#         super(NormDense, self).build(input_shape)

#     def call(self, inputs, **kwargs):
#         norm_w = tf.nn.l2_normalize(self.w, axis=0)
#         norm_inputs = tf.nn.l2_normalize(inputs, axis=1)
#         return tf.matmul(norm_inputs, norm_w)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.units)

#     def get_config(self):
#         config = super(NormDense, self).get_config()
#         config.update({"units": self.units})
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

# class ArcfaceLoss(tf.keras.losses.Loss):
#     def __init__(self, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
#         super(ArcfaceLoss, self).__init__(**kwargs)
#         self.margin1, self.margin2, self.margin3, self.scale = margin1, margin2, margin3, scale
#         self.from_logits, self.label_smoothing = from_logits, label_smoothing
#         self.threshold = np.cos((np.pi - margin2) / margin1)  # grad(theta) == 0
#         self.theta_min = (-1 - margin3) * 2
#         self.batch_labels_back_up = None

#     def build(self, batch_size):
#         self.batch_labels_back_up = tf.Variable(tf.zeros([batch_size], dtype="int64"), dtype="int64", trainable=False)

#     def call(self, y_true, norm_logits):
#         if self.batch_labels_back_up is not None:
#             self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
#         pick_cond = tf.where(tf.math.not_equal(y_true, 0))
#         y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
 
#         if self.margin1 == 1.0 and self.margin2 == 0.0 and self.margin3 == 0.0:
#             theta = y_pred_vals
#         elif self.margin1 == 1.0 and self.margin3 == 0.0:
#             theta = tf.cos(tf.acos(y_pred_vals) + self.margin2)
#         else:
#             theta = tf.cos(tf.acos(y_pred_vals) * self.margin1 + self.margin2) - self.margin3

#         theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)

#         arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale

#         return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

#     def get_config(self):
#         config = super(ArcfaceLoss, self).get_config()
#         config.update(
#             {
#                 "margin1": self.margin1,
#                 "margin2": self.margin2,
#                 "margin3": self.margin3,
#                 "scale": self.scale,
#                 "from_logits": self.from_logits,
#                 "label_smoothing": self.label_smoothing,
#             }
#         )
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
        
# class SupervisedContrastiveLoss(tf.keras.losses.Loss):
#     def __init__(self, temperature=1, name=None):
#         super(SupervisedContrastiveLoss, self).__init__(name=name)
#         self.temperature = temperature

#     def __call__(self, labels, feature_vectors, sample_weight=None):
#         # Normalize feature vectors
#         feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
#         # Compute logits
#         logits = tf.divide(
#             tf.matmul(
#                 feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
#             ),
#             self.temperature,
#         )
#         return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

# with strategy:
#     model =  tf.keras.models.load_model(weight, custom_objects={'resmlp>ChannelAffine': ChannelAffine, 
#                                                                 'NormDense' : NormDense,
#                                                                 'ArcfaceLoss' : ArcfaceLoss,
#                                                                 'SupervisedContrastiveLoss' : SupervisedContrastiveLoss})
#     model.summary()

with strategy:
    encoder = tf.keras.Sequential([
        model.get_layer('input_1'),
        model.get_layer('embedding'),
    ])

    encoder.summary()

    encoder.save(encoder_save)

    encoder = tf.keras.models.load_model(encoder_save)
    encoder.summary()