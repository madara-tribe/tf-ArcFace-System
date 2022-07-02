import os
import json
import numpy as np
import time
from meta_model.cfg import Cfg as csCfg
from meta_model.test import MetaTester
from meta_model.DataLoder import DataLoad as MetaDataLoad

from ArcFace_model.cfg import Cfg
from ArcFace_model.test import Tester

from metrics.cosin_metric import cosin_metric, cosine_similarity

# param
CMed = 0.2
SMed = 0.6
ARCFACE_PATH="ArcFace_model/results/ep96_arcface_model.hdf5"
META_PATH='meta_model/results/ep66_meta_model.hdf5' 
dd = json.load(open("data/cs_label.json"))

# meta model 
cscfg = csCfg
meta_model = MetaTester(cscfg).load_model(META_PATH)

# Arcface
cfg = Cfg
from ArcFace_model.load_model import load_arcface_model as load_pretrain_model
pretrained_model = load_pretrain_model(weights=ARCFACE_PATH)
arcface_model = Tester(cfg).load_arcface_model(pretrained_model)

# database
X, _, y, color_label, shape_label = MetaDataLoad(cscfg).meta_load(valid=False, test=False)
X, y, color_label, shape_label = np.array(X[:400]), np.array(y[:400]), np.array(color_label[:400]), np.array(shape_label[:400])
embed_shape, embed_color = meta_model.predict(X, verbose=1)
X_embed = arcface_model.predict(X, verbose=1)
print(X.shape, y.shape, color_label.shape, shape_label.shape)
print(X_embed.shape, embed_shape.shape, embed_color.shape)


# query
Xqs, _, yqs, color_query, shape_query = MetaDataLoad(cscfg).meta_load(valid=True, test=True)
Xqs, yqs, color_query, shape_query = np.array(Xqs), np.array(yqs), np.array(color_query), np.array(shape_query)
print(Xqs.shape, yqs.shape, color_query.shape, shape_query.shape)

sacc = cacc = xacc = 0
tacc = 0
count = 0
for i, (Xq, yq, yqc, yqs) in enumerate(zip(Xqs, yqs, color_query, shape_query)):
    ####### meta and arcface predict #######
    embed_query_shape, embed_query_color = meta_model.predict(np.expand_dims(Xq, 0), verbose=0)
    X_embed_query = arcface_model.predict(np.expand_dims(Xq, 0), verbose=0)

    # shape
    shape_cossim = [cosin_metric(embed_query_shape, d) for d in embed_shape]
    shape_cossim = np.array(shape_cossim)-SMed
    sidx = np.argmax(shape_cossim)
    pred_shape_idx = shape_label[sidx]
    sacc += 1 if pred_shape_idx==yqs else 0
    # color
    color_cossim = [cosin_metric(embed_query_color, d) for d in embed_color]
    color_cossim = np.array(color_cossim) #*shape_cossim
    cidx = np.argmax(color_cossim)
    pred_color_idx = color_label[cidx]
    cacc += 1 if pred_color_idx==yqc else 0

    # arcface
    cos_sims = [cosin_metric(X_embed_query, d) for d in X_embed]
    cos_sims = np.array(cos_sims) # *shape_cossim
    y_idx = np.argmax(cos_sims)
    pred_idx = y[y_idx]
    xacc += 1 if pred_idx==yq else 0
    #total_sim = np.array(shape_cossim)+ np.array(color_cossim)+ np.array(cos_sims)
    #total_idx = y[np.argmax(total_sim)]
    #tacc += 1 if total_idx==yq else 0
    count +=1

#print("tacc is ", tacc/count)
print("sacc, cacc, xacc is ", sacc/count, cacc/count, xacc/count)

