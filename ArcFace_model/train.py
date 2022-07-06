import sys
#sys.path.append("../")
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from .DataLoder import DataLoad
from .load_model import load_arcface_model
from .random_eraser import get_random_eraser
from tensorflow.keras.utils import to_categorical

class Trainer():
    def __init__(self, config):
        self.cfg = config
        self.loader = DataLoad(config)
        self.eraser = get_random_eraser(v_l=0, v_h=1)
    def train(self, weight_path=None):
        #FROM, TO = 200, 400
        print('train data loading.....')
        X, ys = self.loader.img_load(valid=False, test=False)
        X_val, y_val= self.loader.img_load(valid=True, test=False)
        #X2 = [self.eraser(x) for x in X]
         
        # input image 
        X_, X_val = np.array(X), np.array(X_val)
        y_labels_, y_val = to_categorical(ys, num_classes=122), to_categorical(y_val, num_classes=122)
        
        print(X_.shape, X_val.shape, y_labels_.shape, y_val.shape)
        print(X_.max(), X_.min(), y_labels_.max(), y_val.min()) 
        
        print('model loading.....')
        calllbacks_ = self.loader.create_callbacks()
        model = load_arcface_model(weights=weight_path)
         
        print('start training.....')
        startTime1 = datetime.now()
        hist1 = model.fit(x=[X_, y_labels_], y=y_labels_,
                batch_size=self.cfg.train_batch,
                epochs=self.cfg.epochs, 
                validation_data=([X_val,y_val],  y_val), 
                verbose=1, 
                callbacks=calllbacks_)

        endTime1 = datetime.now()
        diff1 = endTime1 - startTime1
        print("\n")
        print("Elapsed time for Keras training (s): ", diff1.total_seconds())
