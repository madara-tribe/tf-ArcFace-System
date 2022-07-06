import os
import json
import numpy as np
import time
from meta_model.DataLoder import DataLoad as MetaDataLoad
from metrics.cosin_metric import cosin_metric, cosine_similarity

# params
dd = json.load(open("data/cs_label.json"))

class Predictor:
    def __init__(self, cscfg, meta_model, arcface_model):
        self.meta_model = meta_model
        self.arcface_model = arcface_model
        self.cscfg = cscfg

    def load_img_as_base(self):
        # database
        X, _, y, color_label, shape_label = MetaDataLoad(self.cscfg).meta_load(valid=False, test=False)
        X, y, color_label, shape_label = np.array(X), np.array(y), np.array(color_label), np.array(shape_label)
        embed_shape, embed_color = self.meta_model.predict(X, verbose=1)
        X_embed = self.arcface_model.predict(X, verbose=1)
        print(X.shape, y.shape, color_label.shape, shape_label.shape)
        print(X_embed.shape, embed_shape.shape, embed_color.shape)
        return X_embed, embed_shape, embed_color, y, color_label, shape_label

    def load_query_img(self):
        # query
        Xqs, _, yqs, color_query, shape_query = MetaDataLoad(self.cscfg).meta_load(valid=True, test=True)
        Xqs, yqs, color_query, shape_query = np.array(Xqs), np.array(yqs), np.array(color_query), np.array(shape_query)
        X_embed_query = self.arcface_model.predict(Xqs, verbose=1)

        embed_query_shape, embed_query_color = self.meta_model.predict(Xqs, verbose=1)
        print(X_embed_query.shape, yqs.shape, embed_query_shape.shape, embed_query_color.shape)
        return X_embed_query, embed_query_shape, embed_query_color, yqs, color_query, shape_query

    def predict(self, num_candidates=20):
        tacc = count = 0
        # database
        X_embed, embed_shape, embed_color, y, color_label, shape_label = self.load_img_as_base()
        X_embed_query, embed_query_shape, embed_query_color, yqs, color_query, shape_query = self.load_query_img()

        pred_label = []
        start = time.time()
        for i, (yq, yqc, yqs) in enumerate(zip(yqs, color_query, shape_query)):
            ####### meta and arcface predict #######

            # arcface
            cos_sims = cosine_similarity(X_embed_query[i], X_embed)
            X_candidate = [y[idx] for idx in np.argsort(cos_sims)[0][::-1][:num_candidates]]

            # color 
            color_cossim = cosine_similarity(embed_query_color[i], embed_color)
            color_candidate = [color_label[idx] for idx in np.argsort(color_cossim[0])[::-1][:num_candidates]]    
            # shape
            shape_cossim = cosine_similarity(embed_query_shape[i], embed_shape)
            shape_candidate = [shape_label[idx] for idx in np.argsort(shape_cossim[0])[::-1][:num_candidates]]
            
            for x in X_candidate:
                if dd[str(x)]["category"] in shape_candidate and dd[str(x)]["color"] in color_candidate:
                    pred_label.append(x)
            tacc += 1 if yq in pred_label else 0
            count += 1
        print("TF Model Inference Latency is", time.time() - start, "[s]")
        print("tacc is ", tacc/count)

