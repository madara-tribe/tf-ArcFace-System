import os
import numpy as np
import time
from tensorflow.keras.models import Model
from metrics.cosin_metric import cosin_metric, cosine_similarity
from .DataLoder import DataLoad
from .train import MetaTrainer

class MetaTester:
    def __init__(self, config):
        self.loader = DataLoad(config)
        self.cfg = config
    
    def load_model(self, weights): 
        SHAPE_OUTPUT_IDX = -9
        COLOR_OUTPUT_IDX = -10
        model = MetaTrainer(self.cfg).load_model(weights=None)
        model.load_weights(weights)
        
        inputs = model.get_layer(index=0).input
        shape_logits = model.get_layer(index=SHAPE_OUTPUT_IDX).output
        color_logits = model.get_layer(index=COLOR_OUTPUT_IDX).output
        meta_model = Model(inputs=inputs, outputs=[shape_logits, color_logits])
        meta_model.summary()
        return meta_model
    
    def create_holod_vector(self, weight_path):
        # load model
        model = self.load_model(weight_path)
        # prepare embbed data
        X_data, _, _, color_data, shape_data = self.loader.meta_load(valid=False, test=False)
        X_data = np.array(X_data)
        embed_shape, embed_color = model.predict(X_data, verbose=1)
        print(embed_shape.shape, embed_color.shape) # (1138, 256) (1138, 256)
        # prepare query
        X_query, _, _, color_query, shape_query = self.loader.meta_load(valid=True, test=True)
        X_query = np.array(X_query)

        shape_Xh, color_Xh = [], []
        shape_y, color_y = [], []
        shape_th = 0.8
        os.makedirs(save_path, exist_ok=True)
        for i, (Xq, yqc, yqs) in enumerate(zip(X_query, color_query, shape_query)):
            embed_query_shape, embed_query_color = model.predict(np.expand_dims(Xq, 0), verbose=0)
            # color
            color_cossim = [cosin_metric(embed_query_color, d) for d in embed_color]
            cidx = np.argmax(color_cossim)
            cprob = color_cossim[cidx]
            pred_color_idx = color_data[cidx]
            # shape
            shape_cossim = [cosin_metric(embed_query_shape, d) for d in embed_shape]
            sidx = np.argmax(shape_cossim)
            sprob = shape_cossim[sidx]
            pred_shape_idx = shape_data[sidx]

            if pred_shape_idx==yqs and sprob >= shape_th:
                shape_Xh.append(embed_shape[sidx])
                shape_y.append(pred_shape_idx)
            elif pred_color_idx==yqc:
                color_Xh.append(embed_color[cidx])
                color_y.append(pred_color_idx)

        print(len(shape_Xh), len(shape_y), len(color_Xh), len(color_y))
        np.save(save_path + "/shape_Xh", shape_Xh)
        np.save(save_path + "/shape_y", shape_y)
        np.save(save_path + "/color_Xh", color_Xh)
        np.save(save_path + "/color_y", color_y)    


    def test(self, weight_path):
        # load model
        model = self.load_model(weight_path)
        # prepare embbed data
        X_data, _, _, color_data, shape_data = self.loader.meta_load(valid=False, test=False)
        X_data = np.array(X_data)
        embed_shape, embed_color = model.predict(X_data, verbose=1)
        print(embed_shape.shape, embed_color.shape) # (1138, 256) (1138, 256)
        # prepare query
        FROM, TO = 400, 600
        X_query, _, _, color_query, shape_query = self.loader.meta_load(valid=True, test=True)
        X_query = np.array(X_query)[FROM:TO]
        color_query, shape_query = color_query[FROM:TO], shape_query[FROM:TO]
        
        print('testing.....')
        sacc = cacc = pairacc = 0
        start = time.time()
        for i, (Xq, yqc, yqs) in enumerate(zip(X_query, color_query, shape_query)):
            embed_query_shape, embed_query_color = model.predict(np.expand_dims(Xq, 0), verbose=0)
            # color
            color_cossim = [cosin_metric(embed_query_color, d) for d in embed_color]
            pred_color_idx = color_data[np.argmax(color_cossim)]
            # shape
            shape_cossim = [cosin_metric(embed_query_shape, d) for d in embed_shape]
            pred_shape_idx = shape_data[np.argmax(shape_cossim)]

            print(pred_color_idx, yqc, pred_shape_idx, yqs)
            cacc += 1 if pred_color_idx==yqc else 0
            sacc += 1 if pred_shape_idx==yqs else 0
            pairacc += 1 if pred_color_idx==yqc and pred_shape_idx==yqs else 0
        print("TF Model Inference Latency is", time.time() - start, "[s]")
        print("color accuracy", cacc/len(X_query)*100, "%")
        print("shape accuracy", sacc/len(X_query)*100, "%")
        print("color and shape pair accuracy", pairacc/len(X_query)*100, "%")

