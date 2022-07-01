import sys
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from keras_efficientnet_v2.efficientnet_v2 import EfficientNetV2B2
from metrics import archs
from .layers.tf_lambda_network import LambdaLayer
from .cfg import Cfg
from .losses import SoftmaxLoss

#opt = SGD(learning_rate=config.lr, decay=5e-4)
cfg = Cfg
opt = Adam(lr=cfg.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
loss_fn = SoftmaxLoss()

def load_arcface_model(weights=None, use_pretrain=None):
    embed_layer_idx = -4
    lambda_head = 4
    y = Input(shape=(cfg.classes,))
    if use_pretrain:
        model = EfficientNetV2B2(pretrained='imagenet')
    else:
        model = EfficientNetV2B2(pretrained=None) #weights='imagenet')
    inputs = model.get_layer(index=0).input
    x = model.get_layer(index=embed_layer_idx).output
    b, h, g, c =x.shape
    x = LambdaLayer(dim_k=c/lambda_head, r=3, heads=lambda_head, dim_out=c)(x)
    x = GlobalAveragePooling2D()(x)
    outputs = archs.ArcFace(cfg.classes, 30, 0.05)([x, y])
    #outputs = Activation('softmax')(x)
    model = Model(inputs=[inputs,y], outputs=outputs)
    model.compile(optimizer=opt, 
                    loss = loss_fn, #tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])
    if weights:
        print('weight loding ......')
        model.load_weights(weights)
    model.summary()
    return model
