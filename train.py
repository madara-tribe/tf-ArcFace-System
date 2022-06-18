import os, sys
import numpy as np
import cv2
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from DataLoder import DataLoad
from cfg import Cfg
from keras_efficientnet_v2.efficientnet_v2 import EfficientNetV2B2
from layers.tf_lambda_network import LambdaLayer
from metrics import archs, losses


class Trainer():
    def __init__(self, config):
        global HEIGHT, WIDTH
        HEIGHT, WIDTH = config.H, config.W
        self.cfg = config
        self.loader = DataLoad(config)
        #self.opt = SGD(learning_rate=config.lr, momentum=0.9, nesterov=True)
        self.opt = Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    def load_model(self, weights):
        y1 = Input(shape=(self.cfg.classes,))
        model = EfficientNetV2B2(pretrained="imagenet") #(include_top=True, weights='imagenet')
        #model.summary()
        inputs = model.get_layer(index=0).input
        origin = model.get_layer(index=-4).output
        origin = LambdaLayer(dim_k=320, r=3, heads=4, dim_out=1280)(origin)
        x = GlobalMaxPooling2D()(origin)
        x = BatchNormalization()(x)
        #x = archs.ArcFace(self.cfg.classes, 30, 0.05)([x, y1])
        outputs = Dense(self.cfg.classes, activation=None, name="outputs")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.opt, 
                      loss = {"outputs": losses.ArcFaceLoss},
                      metrics=['accuracy'])
        if weights:
            print('weight loding ......')
            model.load_weights(os.path.join(WEIGHT_DIR, weights))
        model.summary()
        return model
        
    def train(self, weight_path=None):
        print('train data loading.....')
        X, y_labels, X_aug = self.loader.img_load(valid=False)
        X_val, y_val, _ = self.loader.img_load(valid=True)
        # input image (cls==128) 
        X_, X_val = np.array(X+X_aug), np.array(X_val)
        y_labels_, y_val = to_categorical(y_labels+y_labels), to_categorical(y_val)
        print(X_.shape, X_val.shape, y_labels_.shape, y_val.shape)
        
        print('model loading.....')
        calllbacks_ = self.loader.create_callbacks() 
        model = self.load_model(weight_path)
       
        print('start training.....')
        startTime1 = datetime.now()
        hist1 = model.fit(x=X_,
                y=y_labels_,
                batch_size=self.cfg.train_batch, 
                epochs=self.cfg.epochs, 
                validation_data=(X_val, y_val), 
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

