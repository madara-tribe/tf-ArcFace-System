import os, sys
import numpy as np
import time
from tensorflow.keras.models import Model
from metrics.cosin_metric import cosin_metric, cosine_similarity
from .DataLoder import DataLoad
from .cfg import Cfg
from .train import MetaTrainer


class MetaTester:
    def __init__(self, config):
        self.loader = DataLoad(config)
        self.cfg = config

    def load_model(self, weights):
        model = MetaTrainer(self.cfg).load_model(weights=None)
        model.load_weights(weights)
        
        inputs = model.get_layer(index=0).input
        shape_logits = model.get_layer(index=-9).output
        color_logits = model.get_layer(index=-10).output
        meta_model = Model(inputs=inputs, outputs=[shape_logits, color_logits])
        meta_model.summary()
        return meta_model
            
    def test(self, weight_path):
        # load model
        model = self.load_model(weight_path)

        # data
        X_data, _, color_data, shape_data = self.loader.meta_load(valid=False, test=False)
        X_data = np.array(X_data)
        embed_shape, embed_color = model.predict(X_data, verbose=1)
        print(embed_shape.shape, embed_color.shape) # (1138, 256) (1138, 256)

        # query
        X_query, y_labels, color_query, shape_query = self.loader.meta_load(valid=True, test=True)
        X_query = np.array(X_query) #[400:600]
        color_query, shape_query = color_query, shape_query
        print(X_query.shape, np.array(y_labels).shape, np.array(color_query).shape, np.array(shape_query).shape)
        print('testing.....')
        sacc = cacc = totalacc = 0
        easy, hard = [], []
        start = time.time()
        for i, (Xq, Xyq, yqc, yqs) in enumerate(zip(X_query, y_labels, color_query, shape_query)):
            embed_query_shape, embed_query_color = model.predict(np.expand_dims(Xq, 0), verbose=0)
            # color
            color_cossim = [cosin_metric(embed_query_color, d) for d in embed_color]
            pred_color_idx = color_data[np.argmax(color_cossim)]
            # shape
            shape_cossim = [cosin_metric(embed_query_shape, d) for d in embed_shape]
            pred_shape_idx = shape_data[np.argmax(shape_cossim)]

            print(pred_color_idx, yqc, pred_shape_idx, yqs)
            if pred_color_idx==yqc and pred_shape_idx==yqs:
                easy.append(Xyq)
                totalacc += 1
            else:
                hard.append(Xyq)
                totalacc += 0 
            #cacc += 1 if pred_color_idx==yqc else 0
            #sacc += 1 if pred_shape_idx==yqs else 0
            #totalacc += 1 if pred_color_idx==yqc and pred_shape_idx==yqs else 0
        print("TF Model Inference Latency is", time.time() - start, "[s]")
        #print("color accuracy", cacc/len(X_query)*100, "%")
        #print("shape accuracy", sacc/len(X_query)*100, "%")
        print("color and shape pair accuracy", totalacc/len(X_query)*100, "%")
        np.save("easy", easy)
        np.save("hard", hard)
