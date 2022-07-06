from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers

import tensorflow as tf
import numpy as np



class ArcFace(Layer):
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False):
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        super(ArcFace, self).__init__()
    
    def get_config(self):
        config = {
            "n_classes" : self.n_classes,
            "s" : self.s,
            "m" : self.m
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.n_classes),
                                      initializer='uniform',
                                      trainable=True)
        super(ArcFace, self).build(input_shape)


    # mainの処理
    def call(self, x):
        y = x[1]
        x_normalize = tf.math.l2_normalize(x[0]) # x = x'/ ||x'||2
        k_normalize = tf.math.l2_normalize(self.kernel) # Wj = Wj' / ||Wj'||2

        cos_m = K.cos(self.m)
        sin_m = K.sin(self.m)
        th = K.cos(np.pi - self.m)
        mm = K.sin(np.pi - self.m) * self.m

        cosine = K.dot(x_normalize, k_normalize) # W.Txの内積
        sine = K.sqrt(1.0 - K.square(cosine))

        phi = cosine * cos_m - sine * sin_m #cos(θ+m)の加法定理

        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)

        else:
            phi = tf.where(cosine > th, phi, cosine - mm)

        output = (y * phi) + ((1.0 - y) * cosine)
        output *= self.s

        return output

    def compute_output_shape(self, input_shape):

        return (input_shape[0][0], self.n_classes)

