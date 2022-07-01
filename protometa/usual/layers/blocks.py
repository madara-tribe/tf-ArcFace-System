from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from tensorflow.keras import layers
import numpy as np


def dw_conv(init, nb_filter, k):
    residual = Conv2D(nb_filter * k, (1, 1), strides=(2, 2), padding='same', use_bias=False)(init)
    residual = BatchNormalization()(residual)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(init)
    x = BatchNormalization()(x)
    x = Activation("swish")(x)
    x = Dropout(0.4)(x)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])
    return x

def res_block(init, nb_filter, k=1):
    x = Activation("swish")(init)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("swish")(x)
    x = Dropout(0.4)(x)
    x = Conv2D(nb_filter * k, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Squeeze_excitation_layer(x)
    x = add([init, x])
    return x

def Squeeze_excitation_layer(input_x):
    ratio = 4
    out_dim = int(np.shape(input_x)[-1])
    squeeze = GlobalAveragePooling2D()(input_x)
    excitation = Dense(units=int(out_dim / ratio))(squeeze)
    excitation = Activation("swish")(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape([-1,1,out_dim])(excitation)
    scale = multiply([input_x, excitation])
    return scale
