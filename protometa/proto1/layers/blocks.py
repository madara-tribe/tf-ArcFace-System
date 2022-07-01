import tensorflow as tf 
from tensorflow.keras.layers import *
import numpy as np

CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
channel_axis = -1
def dw_conv(init, nb_filter, k):
    residual = Conv2D(nb_filter * k, (1, 1), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(init)
    residual = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001)(residual)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(init)
    x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=0.001)(x)
    x = Activation("swish")(x)
    x = Dropout(0.4)(x)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=0.001)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])
    return x

def res_block(init, nb_filter, k=1):
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(init)
    x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=0.001)(x)
    x = Activation("swish")(x)
    x = DepthwiseConv2D(3, padding="same", strides=1, use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=0.001)(x)
    x = Activation("swish")(x)
    x = Squeeze_excitation_layer(x)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=0.001)(x)
    x = Activation("swish")(x)
    x = Dropout(0.2, noise_shape=(None, 1, 1, 1))(x)
    x = add([init, x])
    return x

def Squeeze_excitation_layer(input_x, se_ratio=4):
    h_axis, w_axis = [1, 2]
    filters = input_x.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = filters // se_ratio
    # se = GlobalAveragePooling2D()(inputs)
    # se = Reshape((1, 1, filters))(se)
    se = tf.reduce_mean(input_x, [h_axis, w_axis], keepdims=True)
    se = Conv2D(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER)(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = Activation("swish")(se)
    se = Conv2D(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER)(se)
    se = Activation("sigmoid")(se)
    return Multiply()([input_x, se])

