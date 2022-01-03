import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from UTKload import EPOCHS, BATCH_SIZE, HEIGHT, WIDTH, WEIGHT_DIR, LR, UTKLoad, RACE_NUM_CLS
import keras_efficientnet_v2
from metrics import archs, pca
    
class ArcFace:
    def __init__(self, train_path, val_path, num_race, weight_decay=1e-4):
        self.utkload_ = UTKLoad(gamma=2.0, cosine_annealing=True)
        self.train_path = train_path
        self.val_path = val_path
        self.calllbacks = self.utkload_.create_callbacks()
        self.race_cls = num_race
        self.weight_decay = weight_decay

    def load_arcface_model(self, weights):
        adam = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
        #sgd = SGD(learning_rate=LR, momentum=0.9, nesterov=True)
        input_shape = (HEIGHT, WIDTH, 3)
        y = layers.Input(shape=(self.race_cls,))
        
        model = keras_efficientnet_v2.EfficientNetV2B2(pretrained="imagenet")
        inputs = model.get_layer(index=0).input
        x = model.get_layer(index=-4).output
        x = layers.GlobalMaxPooling2D()(x)
        x = layers.BatchNormalization()(x)
        outputs = archs.ArcFace(n_classes=self.race_cls)([x, y])
        model = Model(inputs=[inputs, y], outputs=outputs)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        if weights:
            print('weight loding ......')
            model.load_weights(os.path.join(WEIGHT_DIR, weights))
        model.summary()
        return model

    def flip(self, x):
        return np.fliplr(x)

    def preprocess(self):
        print('train data loading.....')
        X_train, y_train = self.utkload_.load_data(path=self.train_path, img_size=HEIGHT)
                
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)
        self.X_train, self.y_train = np.array(X_train, dtype='float32')/255, to_categorical(y_train)
        print(self.X_train.shape, self.y_train.shape, self.X_train.max(), self.X_train.min())
        
        print('validation data loading.....')
        #X_val, y_val = self.utkload_.load_data(path=self.val_path, img_size=HEIGHT)
        self.X_val, self.y_val = np.array(X_val, dtype='float32')/255, to_categorical(y_val)
        print(self.X_val.shape, self.y_val.shape, self.X_val.max(), self.X_val.min())
        
    def train(self, weight_path=None):        
        self.preprocess()
        model = self.load_arcface_model(weight_path)

        startTime1 = datetime.now()
        hist1 = model.fit(x=[self.X_train,self.y_train],y=self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([self.X_val, self.y_val], self.y_val), verbose=1, callbacks=self.calllbacks)

        endTime1 = datetime.now()
        diff1 = endTime1 - startTime1
        print("\n")
        print("Elapsed time for Keras training (s): ", diff1.total_seconds())
        print("\n")

        for key in ["loss", "val_loss"]:
            plt.plot(hist1.history[key],label=key)
        plt.legend()

        plt.savefig(os.path.join(WEIGHT_DIR, "model" + str(EPOCHS) + "_training_curves_" + str(WIDTH) + "x" + str(HEIGHT) + ".png"))

        model.save(os.path.join(WEIGHT_DIR, "ep" + str(EPOCHS) + "arcface_model" + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"))
        score = model.evaluate([self.X_val, self.y_val], verbose=0)
        print(score)
        
    def output_embedding(self, weights):
        pca_dim = 2048
        model = self.load_arcface_model(weights=weights)
        inputs_ = model.get_layer(index=0).input
        output_ = model.get_layer(index=-3).output
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
    arcface = None
    weight_path = 'arcface_model_40.hdf5'
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    if arcface:
        train_path, val_path = '../../UTK/UTKFace', None
        arcface_ = ArcFace(train_path, val_path, num_race=RACE_NUM_CLS)
        arcface_.train(weight_path=weight_path)
    else:
        train_path, val_path = '../../UTK/UTKFace', '../../UTK/part3'
        arcface_ = ArcFace(train_path, val_path, num_race=RACE_NUM_CLS)
        arcface_.output_embedding(weights=weight_path)



