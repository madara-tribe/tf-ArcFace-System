import os
import json
import numpy as np
import time
from meta_model.cfg import Cfg as csCfg
from meta_model.test import MetaTester
from meta_model.DataLoder import DataLoad as MetaDataLoad

from ArcFace_model.cfg import Cfg
from ArcFace_model.train import Trainer
from ArcFace_model.test import Tester
from ArcFace_model.DataLoder import DataLoad

from metrics.cosin_metric import cosin_metric, cosine_similarity

# param
shape_cls = 2
color_cls = 11
ARCFACE_PATH="ArcFace_model/weights/ep21_ArcFace.hdf5"
META_PATH='meta_model/weights/ep39_meta_model.hdf5'
path ="data/holdv" 
dd = json.load(open("data/cs_label.json"))


def prepare_meta():
    cscfg = csCfg
    meta_model = MetaTester(cscfg).load_model(META_PATH)
    meta_model.summary()

    Xcs, _, clabel, slabel = MetaDataLoad(cscfg).load_hold_vector(path)
    # query
    X_query, _, color_query, shape_query = MetaDataLoad(cscfg).meta_load(valid=True, test=False)
    X_query = np.array(X_query)[400:600]
    color_query, shape_query = color_query[400:600], shape_query[400:600]
    return meta_model, (np.array(Xcs), clabel, slabel), (X_query, color_query, shape_query)

def prepare_arcface():
    cfg = Cfg
    model = Trainer(cfg).load_model(weights=None)
    arcface_model = Tester(cfg).load_arcface_model(model, ARCFACE_PATH)
    arcface_model.summary()
    X122, Ys122, _, _ = DataLoad(cfg).load_hold_vector(path)

    X_query, y_query, _ = DataLoad(cfg).img_load(valid=True, test=False)
    X_query, y_query = np.array(X_query[600:800]), y_query[600:800]
    return arcface_model, (np.array(X122), Ys122), (X_query, y_query)

def creating_meta_embbeding(meta_model, X_meta_datas, meta_querys):
    print("creating meta embbeding")
    color_cossims, shape_cossims = [], []
    embed_shape, embed_color = meta_model.predict(X_meta_datas[0], verbose=1)
    for i, Xq in enumerate(meta_querys[0]):
        embed_query_shape, embed_query_color = meta_model.predict(np.expand_dims(Xq, 0), verbose=0)
        color_cossim = [cosin_metric(embed_query_color, d) for d in embed_color]
        shape_cossim = [cosin_metric(embed_query_shape, d) for d in embed_shape]
        color_cossims.append(color_cossim)
        shape_cossims.append(shape_cossim)
    return color_cossims, shape_cossims

def creating_arcface_embbeding(arcface_model, X_arc_datas, arc_querys):
    print("creating arcface embbeding")
    arcface_cossims= []
    arcface_embed = arcface_model.predict(X_arc_datas[0], verbose=1)
    for i, Xq in enumerate(X_arc_datas[0]):
        embed_query = arcface_model.predict(np.expand_dims(Xq, 0), verbose=0)
        cossim = [cosin_metric(embed_query, d) for d in arcface_embed]
        arcface_cossims.append(cossim)
    return arcface_cossims


# arcface
meta_model, X_meta_datas, meta_querys = prepare_meta()
print(X_meta_datas[0].shape, np.array(X_meta_datas[1]).max(), np.array(X_meta_datas[2]).max())
# cs
arcface_model, X_arc_datas, arc_querys = prepare_arcface()
print(X_arc_datas[0].shape, np.array(X_arc_datas[1]).max())
#(122, 260, 260, 3) (122, 256, 256, 3) 121 10 1
start = time.time()
color_cossims, shape_cossims = creating_meta_embbeding(meta_model, X_meta_datas, meta_querys)
print(np.array(color_cossims).shape, np.array(shape_cossims).shape)
#X_arcs = arcface_model.predict(X_arc_datas[0])
np.save("data/color_cossims", color_cossims)
np.save("data/shape_cossims", shape_cossims)
    
arcface_cossims = creating_arcface_embbeding(arcface_model, X_arc_datas, arc_querys)
print(np.array(arcface_cossims).shape)
np.save("data/arcface_cossims", arcface_cossims)

acc = count = 0
for arcface, color, shape, y_label in zip(arcface_cossims, color_cossims, shape_cossims, arc_querys[1]):
    color, shape = np.array(color), np.array(shape)
    confs = (shape/shape_cls)+(color/color_cls)
    pred_idx = np.argmax([conf+val for conf, val in zip(confs, arcface)])
    print(count, pred_idx, y_label)
    acc += 1 if pred_idx==y_label else 0
    count += 1
print("total arcface accuracy with meta inf is {} %", acc/len(arcface_cossims)*100)
print("TF Model Inference Latency is", time.time() - start, "[s]")
