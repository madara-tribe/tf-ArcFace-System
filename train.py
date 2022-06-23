import os, sys
import numpy as np
import cv2
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from DataLoder import DataLoad
from cfg import Cfg
from keras_efficientnet_v2.efficientnet_v2 import EfficientNetV2B2
from layers.tf_lambda_network import LambdaLayer
from metrics import archs


class Trainer():
    def __init__(self, config):
        self.cfg = config
        self.loader = DataLoad(config)
        #self.opt = SGD(learning_rate=config.lr, decay=5e-4)
        self.opt = Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    def load_model(self, weights):
        y = Input(shape=(self.cfg.classes,))
        model = EfficientNetV2B2(pretrained=None) #(include_top=True, weights='imagenet')
        inputs = model.get_layer(index=0).input
        x = model.get_layer(index=-4).output
        x = LambdaLayer(dim_k=320, r=3, heads=4, dim_out=1280)(x)
        x = GlobalAveragePooling2D()(x)
        #x = Dense(self.cfg.classes, activation = 'linear')(x)
        #x = Lambda(lambda x: K.l2_normalize(x,axis=1))(x)
        x = archs.ArcFace(self.cfg.classes, 30, 0.05)([x, y])
        outputs = Activation('softmax')(x)
        model = Model(inputs=[inputs,y], outputs=outputs)
        model.compile(optimizer=self.opt, 
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(),
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
        X_val, y_val = X_val[200:400], y_val[200:400]
        # input image (cls==128) 
        X_, X_val = np.array(X+X_aug), np.array(X_val)
        y_labels_, y_val = np.array(y_labels+y_labels), np.array(y_val)
        #y_labels_, y_val = to_categorical(y_labels+y_labels, num_classes=self.cfg.classes, dtype='uint8'), to_categorical(y_val, num_classes=self.cfg.classes, dtype='uint8')
        #print(X_.shape, X_val.shape, y_labels_.shape, y_val.shape)
        #print(X_.max(), X_.min()) 
        print('model loading.....')
        calllbacks_ = self.loader.create_callbacks() 
        model = self.load_model(weight_path)
       
        print('start training.....')
        startTime1 = datetime.now()
        hist1 = model.fit(x=[X_,y_labels_],
                y=y_labels_,
                batch_size=self.cfg.train_batch, 
                epochs=self.cfg.epochs, 
                validation_data=([X_val,y_val],  y_val), 
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

