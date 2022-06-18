import os, sys
import numpy as np
import cv2, json
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from DataLoder import DataLoad
from cfg import Cfg
from keras_efficientnet_v2.efficientnet_v2 import EfficientNetV2B2
from layers.tf_lambda_network import LambdaLayer
from metrics import archs


def cal_acc(y_true, y_pred, argmax_idx):
    if y_pred==y_true or y_true==argmax_idx:
        acc = 1
    else:
        acc = 0
    return acc

        
def load_model(cfg, weights):
    y1 = Input(shape=(cfg.classes,))
    yc = Input(shape = (11,))
    ys = Input(shape=(2,))
    
    model = EfficientNetV2B2(pretrained="imagenet") #(include_top=True, weights='imagenet')
    #model.summary()
    inputs = model.get_layer(index=0).input
    origin = model.get_layer(index=-4).output
    #origin = LambdaLayer(dim_k=320, r=3, heads=4, dim_out=1280)(origin)
    x = GlobalMaxPooling2D()(origin)
    x = BatchNormalization()(x)
    #x = archs.ArcFace(self.cfg.classes, 30, 0.05)([x, y1])
    outputs = Dense(cfg.classes, activation=None, name="outputs")(x)

    # color meta
    cx = LambdaLayer(dim_k=320, r=3, heads=4, dim_out=1280)(origin)
    cx = GlobalMaxPooling2D()(cx)
    cx = BatchNormalization()(cx)
    color_logits = Dense(11, activation='softmax', name='color_logits')(cx)

    # shape meta
    sx = LambdaLayer(dim_k=320, r=3, heads=4, dim_out=1280)(origin)
    sx = GlobalMaxPooling2D()(sx)
    sx = BatchNormalization()(sx)
    shape_logits = Dense(2, activation='softmax', name="shape_logits")(sx)

    model = Model(inputs=inputs, outputs=[outputs, color_logits, shape_logits])
    if weights:
        print('weight loding ......')
        model.load_weights(weights)
    model.summary()
    return model
        
def evaluate(cfg, weight_path):
    loader = DataLoad(cfg)
    X_val, _, _, y_val, vc_label, vs_label = loader.load_data(valid=True)
    # input image (cls==128) 
    X_val = np.array(X_val)
    print('model loading.....')
    model = load_model(cfg, weight_path)

    print('evaluating.....')
    th = pred_idx = acc = 0
    dd = json.load(open("cs_label.json"))
    
    pred_X, pred_color, pred_shape = model.predict(X_val, verbose=1)
    for i, (Xp, c, s) in enumerate(zip(pred_X, pred_color, pred_shape)):
        clabel, slabel = np.argmax(c), np.argmax(s)
        for j, (k, v) in enumerate(dd.items()):
            if v['category']==slabel or v['color']==clabel:
                if Xp[j]>th:
                    th = Xp[j]
                    pred_idx = j
        argmax_idx = np.argmax(Xp)
        #print("i, len(Xp), max(Xp), th, Xp[pred_idx], pred_idx, np.array(y_val).shape")
        #print(i, len(Xp), max(Xp), th, Xp[pred_idx], pred_idx, y_val[i], np.array(y_val).shape)
        #print("Xp", Xp)
        acc += cal_acc(y_val[i], pred_idx, argmax_idx)
    print("total acccuracy is ", acc/len(X_val))
        
    
        
if __name__=='__main__':
    idx = int(sys.argv[1])
    cfg = Cfg
    weight_path = "weights/arcface_model_{}.hdf5".format(idx)
    evaluate(cfg, weight_path)
