import sys
#sys.path.append("../")
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from .DataLoder import DataLoad
from .load_model import load_arcface_model


class Trainer():
    def __init__(self, config):
        self.cfg = config
        self.loader = DataLoad(config)
        
    def train(self, weight_path=None):
        #FROM, TO = 200, 400
        print('train data loading.....')
        X, ys = self.loader.img_load(valid=False, test=False)
        X_val, y_val= self.loader.img_load(valid=True, test=False)
  
        # input image 
        X_, X_val = np.array(X), np.array(X_val)
        y_labels_, y_val = np.array(ys), np.array(y_val)
        print(X_.shape, X_val.shape, y_labels_.shape, y_val.shape)
        print(X_.max(), X_.min(), y_labels_.max(), y_val.min()) 
        print('model loading.....')
        calllbacks_ = self.loader.create_callbacks()
        model = load_arcface_model(weights=weight_path)
         
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
