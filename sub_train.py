import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from DataLoder import DataLoad
from cfg import Cfg
from layers.resnet import create_model
from layers.tf_lambda_network import LambdaLayer
from metrics import archs


class Trainer():
    def __init__(self, config):
        global HEIGHT, WIDTH
        HEIGHT, WIDTH = config.H, config.W
        self.cfg = config
        self.loader = DataLoad(config)
        #self.opt = SGD(learning_rate=config.lr, momentum=0.9, nesterov=True)
        self.opt = Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    def load_model(self, weights):
        model = create_model(input_shape=(HEIGHT, WIDTH,3),k=1, lr=1e-3)
        model.compile(optimizer=self.opt, 
                      loss = {"color_logits": "categorical_crossentropy",
                              "shape_logits": "binary_crossentropy"},
                      metrics=['accuracy'])
        if weights:
            print('weight loding ......')
            model.load_weights(os.path.join(WEIGHT_DIR, weights))
        model.summary()
        return model
        
    def train(self, weight_path=None):
        print('train data loading.....')
        X, X_aug, x_colors, x_shapes, color_label, shape_label = self.loader.meta_load(valid=False)
        X_val, _, val_colors, val_shapes, vc_label, vs_label = self.loader.meta_load(valid=True)
        # input image (cls==128) 
        X, X_val = np.array(X+X_aug), np.array(X_val)
        
        # color meta (cls==11)
        x_colors, val_colors = np.array(x_colors + x_colors), np.array(val_colors)
        color_label, vc_label = to_categorical(color_label+color_label), to_categorical(vc_label)
        # shape meta (cls==2)
        x_shapes, val_shapes = np.array(x_shapes+x_shapes), np.array(val_shapes)
        shape_label, vs_label = to_categorical(shape_label+shape_label), to_categorical(vs_label)

        print('model loading.....')
        calllbacks_ = self.loader.create_callbacks() 
        model = self.load_model(weight_path)
       
        print('start training.....')
        startTime1 = datetime.now()
        hist1 = model.fit(x=X,
                y=[color_label, shape_label],
                batch_size=self.cfg.train_batch, 
                epochs=self.cfg.epochs, 
                validation_data=(X_val, [vc_label, vs_label]), 
                verbose=1, 
                callbacks=calllbacks_)

        endTime1 = datetime.now()
        diff1 = endTime1 - startTime1
        print("\n")
        print("Elapsed time for Keras training (s): ", diff1.total_seconds())
        
if __name__=='__main__':
    cfg = Cfg
    weight_path = None
    os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
    arcface = Trainer(cfg)
    arcface.train(weight_path=weight_path)

