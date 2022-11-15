import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, regularizers, backend as K
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Activation, Conv2D, Input, GlobalAveragePooling2D, Concatenate, InputLayer, \
ReLU, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, Softmax, Lambda, LeakyReLU

class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, **kwargs):
        super(NormDense, self).__init__(**kwargs)
        self.init = keras.initializers.glorot_normal()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name="norm_dense_w", shape=(input_shape[-1], self.units), initializer=self.init, trainable=True)
        super(NormDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        norm_w = tf.nn.l2_normalize(self.w, axis=0)
        norm_inputs = tf.nn.l2_normalize(inputs, axis=1)
        return tf.matmul(norm_inputs, norm_w)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(NormDense, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)