import os, sys
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from .DataLoder import DataLoad
from .cfg import Cfg
from .layers.resnet import create_model

class MetaTrainer():
    def __init__(self, config):
        global HEIGHT, WIDTH
        HEIGHT, WIDTH = config.H, config.W
        self.cfg = config
        self.loader = DataLoad(config)
        #self.opt = SGD(learning_rate=config.lr, momentum=0.9, nesterov=True)
        self.opt = Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    def load_model(self, weights):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        model = create_model(HEIGHT, WIDTH, k=1, lr=1e-3)
        model.compile(optimizer=self.opt, 
                      loss = {"color_logits": loss_fn,
                              "shape_logits": loss_fn},
                      metrics=['accuracy'])
        if weights:
            print('weight loding ......')
            model.load_weights(weights)
        model.summary()
        return model
        
    def train(self, weight_path=None):
        FROM, TO = 200, 400
        print('train data loading.....')
        X, X_aug, _, color_label, shape_label = self.loader.meta_load(valid=False)
        X_val, _, _, vc_label, vs_label = self.loader.meta_load(valid=True)
        X_val, vc_label, vs_label = X_val[FROM:TO], vc_label[FROM:TO], vs_label[FROM:TO]
        # input image (cls==128) 
        X, X_val = np.array(X+X_aug), np.array(X_val)
        
        # color meta (cls==11)
        color_label, vc_label = np.array(color_label+color_label), np.array(vc_label)
        # shape meta (cls==2)
        shape_label, vs_label = np.array(shape_label+shape_label), np.array(vs_label)
        print(X.shape, X_val.shape, color_label.shape, shape_label.shape, X.min(), X.max(), vc_label.shape, vs_label.shape)
        print('model loading.....')
        calllbacks_ = self.loader.create_callbacks() 
        model = self.load_model(weight_path)
       
        print('start training.....')
        startTime1 = datetime.now()
        hist1 = model.fit(x=[X, color_label, shape_label],
                y=[color_label, shape_label],
                batch_size=self.cfg.train_batch, 
                epochs=self.cfg.epochs, 
                validation_data=([X_val, vc_label, vs_label], [vc_label, vs_label]), 
                verbose=1, 
                callbacks=calllbacks_)

        endTime1 = datetime.now()
        diff1 = endTime1 - startTime1
        print("\n")
        print("Elapsed time for Keras training (s): ", diff1.total_seconds())
        
