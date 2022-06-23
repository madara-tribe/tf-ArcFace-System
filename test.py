import os, sys
import time
import numpy as np
from tensorflow.keras.models import Model
from DataLoder import DataLoad
from cfg import Cfg 
from layers.resnet import create_model


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cosine_similarity(q, h):
    if q.ndim == 1:
        q = q.reshape(1, -1)
    q_norm = np.linalg.norm(q, axis=1)
    h_norm = np.linalg.norm(h, axis=1)
    cosine_sim = np.dot(q, h.T)/(q_norm*h_norm+1e-10)
    return cosine_sim

        
class Tester:
    def __init__(self, cfg):
        self.loader = DataLoad(cfg)

    def load_embbed_model(self, weights):
        model = create_model(256, 256, k=1, lr=1e-3)    
        model.load_weights(weights)
        #model.summary()
        model.load_weights(weights)
        
        # embbed model
        embed_inputs = model.get_layer(index=0).input
        emmbed_shape = model.get_layer(index=-5).output
        emmbed_color = model.get_layer(index=-6).output
        embbed_model = Model(inputs=embed_inputs, outputs=[emmbed_shape, emmbed_color])
        embbed_model.summary()
        return embbed_model
    
    def load_querys(self):
        path = "data/holdv"
        X_data, _, cy_data, sy_data = self.loader.load_hold_vector(path)
        X_data = np.array(X_data)
        X_query, _, cy_query, sy_query = self.loader.meta_load(valid=True)
        X_query, cy_query, sy_query = np.array(X_query), np.array(cy_query), np.array(sy_query)
        X_query, cy_query, sy_query= X_query[600], cy_query[600], sy_query[600]
        if len(X_query.shape)==3:
            X_query = np.expand_dims(X_query, 0)
        return (X_data, cy_data, sy_data), (X_query, cy_query, sy_query)
        
        

    def test(self, weight_path):
        datas, qyerys = self.load_querys()
        model = self.load_embbed_model(weight_path)
        # prpare
        X_shape, X_color = model.predict(datas[0], verbose=1)
        
        sacc = cacc = 0
        start = time.time()
        q_shape, q_color = model.predict(qyerys[0], verbose=1)
        print(X_shape.shape, X_color.shape, q_shape.shape, q_color.shape)

        shape_sims = [cosin_metric(q_shape, d) for d in X_shape]
        color_sims = [cosin_metric(q_color, d) for d in X_color]
        #np.save("npy/color_sim600", color_sims)
        #np.save("npy/shape_sim600", shape_sims)
        
        print(max(datas[1]), max(datas[2]))
        shape_idx = datas[2][np.argmax(shape_sims)]
        color_idx = datas[1][np.argmax(color_sims)]
        print(shape_idx, color_idx, qyerys[2], qyerys[1])
        if shape_idx==qyerys[2]:
            sacc += 1
        if color_idx==qyerys[1]:
            cacc += 1
    
        print("TF Model Inference Latency is", time.time() - start, "[s]")
        print('color shape accuracy is {} {}'.format(cacc*100, sacc*100))
      

if __name__=='__main__':
    weight_path = "weights/arcface_model_30.hdf5"
    cfg = Cfg
    tester = Tester(cfg)
    tester.test(weight_path)


