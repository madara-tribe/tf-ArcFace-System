import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from DataLoder import DataLoad
from cfg import Cfg
#from tensorflow.keras.applications.efficientnet import EfficientNetB1
from keras_efficientnet_v2.efficientnet_v1 import EfficientNetV1B2
from layers.tf_lambda_network import LambdaLayer
from metrics import archs, pca

class Trainer():
    def __init__(self, config):
        global HEIGHT, WIDTH
        HEIGHT, WIDTH = config.H, config.W
        self.cfg = config
        self.loader = DataLoad(config)
        self.opt = SGD(learning_rate=config.lr, momentum=0.9, nesterov=True)

    def load_model(self, weights):
        input_shape = (HEIGHT, WIDTH, 3)
        color_ = (HEIGHT, WIDTH, 11)
        shape_ = (HEIGHT, WIDTH, 2)
        y1 = layers.Input(shape=(self.cfg.classes,))
        yc = layers.Input(shape = (11,))
        ys = layers.Input(shape=(2,))
        
        model = EfficientNetV1B2(pretrained="imagenet") #(include_top=True, weights='imagenet')
        #model.summary()
        inputs = model.get_layer(index=0).input
        origin = model.get_layer(index=-4).output
        #x = LambdaLayer(dim_k=320, r=3, heads=4, dim_out=1280)(x)
        x = layers.GlobalMaxPooling2D()(origin)
        x = layers.BatchNormalization()(x)
        x = archs.ArcFace(self.cfg.classes, 30, 0.05)([x, y1])
        outputs = layers.Activation('softmax')(x)

        # color meta
        cx = layers.GlobalMaxPooling2D()(origin)
        cx = layers.BatchNormalization()(cx)
        cx = layers.Dense(11, activation=None)(cx)
        color_logits = layers.Activation('softmax')(cx)

        # shape meta
        sx = layers.GlobalMaxPooling2D()(origin)
        sx = layers.BatchNormalization()(sx)
        sx = layers.Dense(2, activation=None)(sx)
        shape_logits = layers.Activation('softmax')(sx)

        model = Model(inputs=[inputs, y1], outputs=[outputs, color_logits, shape_logits])
        model.compile(optimizer=self.opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        ##if weights:
        print('weight loding ......')
        #    model.load_weights(os.path.join(WEIGHT_DIR, weights))
        model.summary()
        return model
        
    def train(self, weight_path=None):
        print('train data loading.....')
        X, x_color, x_shapes, y_labels, color_label, shape_label = self.loader.load_data(valid=False)
        X_val, val_color, val_shapes, y_val, vcolor_label, vshape_label = self.loader.load_data(valid=True)
        
        # input image (cls==128) 
        X, X_val, y_labels, y_val = np.array(X), np.array(X_val), np.array(y_labels), np.array(y_val)
        y_labels, y_val = to_categorical(y_labels), to_categorical(y_val)
        print(X.max(), X.min(), y_labels.max(), y_val.max())
        print(X.shape, X_val.shape, y_labels.shape, y_val.shape)
        # color meta (cls==11)
        x_color, color_label, val_color, vcolor_label = np.array(x_color), np.array(color_label), np.array(val_color), np.array(vcolor_label) 
        color_label, vcolor_label = to_categorical(color_label), to_categorical(vcolor_label)

        # shape meta (cls==2)
        x_shapes, val_shapes, shape_label, vshape_label = np.array(x_shapes), np.array(val_shapes), np.array(shape_label), np.array(vshape_label) 
        shape_label, vshape_label = to_categorical(shape_label), to_categorical(vshape_label)


        calllbacks_ = self.loader.create_callbacks() 
        model = self.load_model(weight_path)
       

 
        startTime1 = datetime.now()
        hist1 = model.fit(x=[X, y_labels] ,
                y=[y_labels,color_label, shape_label],
                batch_size=self.cfg.train_batch, 
                epochs=self.cfg.epochs, 
                validation_data=([X_val,y_val], [y_val, vcolor_label, vshape_label]), 
                verbose=1, 
                callbacks=calllbacks_)

        endTime1 = datetime.now()
        diff1 = endTime1 - startTime1
        print("\n")
        print("Elapsed time for Keras training (s): ", diff1.total_seconds())

        #plt.savefig(os.path.join("model" + str(EPOCHS) + "_training_curves_" + str(WIDTH) + "x" + str(HEIGHT) + ".png"))
        #model.save(os.path.join(WEIGHT_DIR, "ep" + str(EPOCHS) + "arcface_model" + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"))
        #score = model.evaluate([self.X_val, self.y_val], verbose=0)
        #print(score)
        
    def output_embedding(self, weights):
        pca_dim = 2048
        model = self.load_arcface_model(weights=weights)
        inputs_ = model.get_layer(index=0).input
        output_ = model.get_layer(index=-5).output
        print(output_.shape, output_)
        predict_model = Model(inputs=inputs_, outputs=output_)
        predict_model.summary()
        
        self.preprocess()
        test_label = [np.argmax(y_) for y_ in self.y_train]
        embedding = predict_model.predict(self.X_train, verbose=1)
        if embedding.shape[1]>pca_dim:
            embedding = pca.pca_(embedding, dim=pca_dim)
        np.save(os.path.join(WEIGHT_DIR, 'X_embedding'), embedding)
        np.save(os.path.join(WEIGHT_DIR, 'y_embedding'), test_label)
        
if __name__=='__main__':
    cfg = Cfg
    weight_path = None
    os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
    
    arcface = Trainer(cfg)
    arcface.train(weight_path=weight_path)

    
    #arcface_ = ArcFace(train_path, val_path, num_race=RACE_NUM_CLS)
    #arcface_.output_embedding(weights=weight_path)
