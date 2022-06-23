import os, sys
import time
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from DataLoder import DataLoad
from cfg import Cfg
from keras_efficientnet_v2.efficientnet_v2 import EfficientNetV2B2
from layers.tf_lambda_network import LambdaLayer
from metrics import archs
from metrics.cosin_metric import cosin_metric, cosine_similarity

path = "../center/holdv"

N = -4
class Tester:
    def __init__(self, config):
        self.loader = DataLoad(config)
        self.cfg = config

    def load_arcface_model(self, weights):
        num_cls = self.cfg.classes
        y = Input(shape=(num_cls,))
        model = EfficientNetV2B2(pretrained=None) #(include_top=True, weights='imagenet')
        inputs = model.get_layer(index=0).input
        x = model.get_layer(index=-4).output
        x = LambdaLayer(dim_k=320, r=3, heads=4, dim_out=1280)(x)
        x = GlobalAveragePooling2D()(x)
        #x = Dense(self.cfg.classes, activation = 'linear')(x)
        #x = Lambda(lambda x: K.l2_normalize(x,axis=1))(x)
        x = archs.ArcFace(num_cls, 30, 0.05)([x, y])
        outputs = Activation('softmax')(x)
        models = Model(inputs=[inputs,y], outputs=outputs)
        models.load_weights(weights)
        
        # arcface model
        embed_inputs = models.get_layer(index=0).input
        embed_out = models.get_layer(index=N).output
        arcface_model = Model(inputs=embed_inputs, outputs=embed_out)
        arcface_model.summary()
        return arcface_model
    
    def load_querys(self):
        #X_data, y_data, _ = self.loader.img_load(valid=None)
        X_data, y_data, _, _ = self.loader.load_hold_vector(path)
        X_data = np.array(X_data)
        X_query, y_query, _ = self.loader.img_load(valid=True)
        X_query, y_query = X_query[600], y_query[600]
        if len(X_query.shape)==3:
            X_query = np.expand_dims(X_query, 0)
        return X_data, y_data, X_query, y_query
        
        

    def test(self, weight_path):
        X_data, y_data, X_query, y_query = self.load_querys()
        model = self.load_arcface_model(weight_path)
        # prpare
        X_data = model.predict(X_data, verbose=1)

        acc = 0
        start = time.time()
        X_query = model.predict(X_query, verbose=1)
        print(X_data.shape, X_query.shape)
        cos_sims = [cosin_metric(X_query, d) for d in X_data]
        np.save("cossim600", cos_sims)
        max_idx = np.argmax(cos_sims)
        pred_idx = y_data[max_idx]
        print(max_idx, pred_idx, y_query)
        if pred_idx==y_query:
            acc += 1
        else:
            acc += 0
        print("TF Model Inference Latency is", time.time() - start, "[s]")
        print('accuracy is {}'.format(acc/len(X_query)*100))


if __name__=='__main__':
    cfg = Cfg
    weight_path = "weights/arcface_model_29.hdf5"
    os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
    tester = Tester(cfg)
    tester.test(weight_path)


