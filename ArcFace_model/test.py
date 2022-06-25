import os
import time
import numpy as np
from tensorflow.keras.models import Model
from metrics.cosin_metric import cosin_metric, cosine_similarity
from .layers.tf_lambda_network import LambdaLayer
from .DataLoder import DataLoad
from .cfg import Cfg
from .train import Trainer

N = -3
class Tester:
    def __init__(self, config):
        self.loader = DataLoad(config)
        self.cfg = config

    def load_arcface_model(self, model, weights):
        
        model.load_weights(weights)
        # arcface model
        embed_inputs = model.get_layer(index=0).input
        embed_out = model.get_layer(index=N).output
        arcface_model = Model(inputs=embed_inputs, outputs=embed_out)
        arcface_model.summary()
        return arcface_model
    
    def load_querys(self):
        X_data, y_data, _ = self.loader.img_load(valid=False, test=False)
        #X_data, y_data, _, _ = self.loader.load_hold_vector(path)
        X_data = np.array(X_data)
        X_query, y_query, _ = self.loader.img_load(valid=True, test=False)
        X_query, y_query = np.array(X_query[600:800]), y_query[600:800]
        return X_data, y_data, X_query, y_query
        
        

    def test(self, weight_path):
        model = Trainer(self.cfg).load_model(weights=None)
        X_data, y_data, X_query, y_query = self.load_querys()
        model = self.load_arcface_model(model, weight_path)
        # prpare
        X_embbed_data = model.predict(X_data, verbose=1)

        acc = 0
        start = time.time()
        for Xq, yq in zip(X_query, y_query):
            embbed_query = model.predict(np.expand_dims(Xq, 0), verbose=0)
            cos_sims = [cosin_metric(embbed_query, d) for d in X_embbed_data]
            pred_idx = y_data[np.argmax(cos_sims)]
            print(pred_idx, yq)
            acc += 1 if pred_idx==yq else 0
        print("TF Model Inference Latency is", time.time() - start, "[s]")
        print('accuracy is {} %'.format(acc/len(X_query)*100))

