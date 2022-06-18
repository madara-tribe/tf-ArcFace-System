from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers

import tensorflow as tf
import numpy as np


n_classes=122
s=30
m=0.50
easy_margin=False

def ArcFaceLoss(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    y = y_true
    kernel = 1
    x_normalize = tf.math.l2_normalize(y_pred) # x = x'/ ||x'||2
    #k_normalize = tf.math.l2_normalize(kernel) # Wj = Wj' / ||Wj'||2
    
    cos_m = K.cos(m)
    sin_m = K.sin(m)
    th = K.cos(np.pi - m)
    mm = K.sin(np.pi - m) * m

    cosine = x_normalize * kernel #K.dot(x_normalize, k_normalize) # W.Txの内積
    sine = K.sqrt(1.0 - K.square(cosine))

    phi = cosine * cos_m - sine * sin_m #cos(θ+m)の加法定理

    if easy_margin:
        phi = tf.where(cosine > 0, phi, cosine)

    else:
        phi = tf.where(cosine > th, phi, cosine - mm)

    # 正解クラス:cos(θ+m) 他のクラス:cosθ
    output = (y_true * phi) + ((1.0 - y_true) * cosine)
    output *= s
    final_pred = tf.nn.softmax(output)
    loss_val = cce(y_true, final_pred)
    return loss_val

    
