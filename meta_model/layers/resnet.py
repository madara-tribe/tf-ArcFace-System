import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from .blocks import *
from .tf_lambda_network import LambdaLayer
from .archs import ArcFace

def create_model(h, w, k=1, lr=1e-3):
    lambda_heads = 4
    color_cls = 11
    shape_cls = 2

    inputs = Input(shape=(h, w, 3))
    i = 0
    nb_filter = [16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16]

    # 0
    x0 = first_conv(inputs, nb_filter, i, k=1) 
    i += 1

    #1
    x = dw_conv(x0, nb_filter[i], k)
    x = res_block(x, k, nb_filter[i])
    x1 = res_block(x, k, nb_filter[i])
    i += 1

    #2
    x = dw_conv(x1, nb_filter[i], k)
    x = res_block(x, k, nb_filter[i])
    x2 = res_block(x, k, nb_filter[i])
    i += 1

    #3
    x = dw_conv(x2, nb_filter[i], k)
    x = res_block(x, k, nb_filter[i])
    x3 = res_block(x, k, nb_filter[i])
    i += 1

    #4
    x = dw_conv(x3, nb_filter[i], k)
    x = res_block(x, k, nb_filter[i])
    x4 = res_block(x, k, nb_filter[i])

    b, g, f, c = x4.shape
    cxlambda = sxlambda = LambdaLayer(dim_k=c/lambda_heads, r=3, heads=lambda_heads, dim_out=c)(x4)
    # color
    cy = Input(shape=(color_cls,), name='color_label')
    cx = GlobalAveragePooling2D()(cxlambda)
    cx = BatchNormalization()(cx)
    cx = ArcFace(color_cls, 30, 0.05)([cx, cy])
    cx = Activation("softmax", name='color_logits')(cx)

    # shape
    sy = Input(shape=(shape_cls,), name='shape_label')
    sx = GlobalAveragePooling2D()(sxlambda)
    sx = BatchNormalization()(sx)
    sx = ArcFace(shape_cls, 30, 0.05)([sx, sy])
    sx = Activation("softmax", name='shape_logits')(sx)

    model = Model(inputs=[inputs, cy, sy], outputs=[cx, sx])
    return model


