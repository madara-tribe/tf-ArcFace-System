import sys
sys.path.append("../")
sys.path.append("../../")
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from keras_efficientnet_v2.efficientnet_v2 import EfficientNetV2B2
from layers import archs, blocks
from layers.tf_lambda_network import LambdaLayer
from cfg import Cfg

#opt = SGD(learning_rate=config.lr, decay=5e-4)
cfg = Cfg
opt = Adam(lr=cfg.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
ARCFACE_USE = True if cfg.classes<40 else None
print("Arcface is ", ARCFACE_USE)


def load_arcface_model(weights=None):
    embed_layer_idx = -4
    lambda_head = 4
    y = Input(shape=(cfg.classes,))
    model = EfficientNetV2B2(pretrained='imagenet')
    inputs = model.get_layer(index=0).input
    x = model.get_layer(index=embed_layer_idx).output
    b, h, g, c =x.shape
    x = blocks.mbconv_block(x, c)
    x = LambdaLayer(dim_k=c/lambda_head, r=3, heads=lambda_head, dim_out=c)(x)
    logits = GlobalAveragePooling2D()(x)
    #if ARCFACE_USE is not None:
        #logits = archs.ArcFace(cfg.classes, 30, 0.05)([logits, y])
    outputs = Dense(cfg.classes, activation='softmax')(logits)
    model = Model(inputs=[inputs,y], outputs=outputs)
    model.compile(optimizer=opt, 
                    loss = "categorical_crossentropy",
                    metrics=['accuracy'])
    if weights:
        print('weight loding ......')
        model.load_weights(weights)
    model.summary()
    return model
