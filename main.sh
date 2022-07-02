#!/bin/sh
ARC_WIGHT_PATH="ArcFace_model/weights/arcface_model_96.hdf5"
META_EVAL_WEIGHT_PATH='meta_model/results/ep66_meta_model.hdf5'
META_TRAIN_WEIGHT_PATH='meta_model/hdf5/ep25_meta2.hdf5'
python3 main.py \
        --main \
        --arcface \
        --eval \
        --arcface_model_weight_path $ARC_WIGHT_PATH \
   #--arcface_model_weight_path $ARC_WIGHT_PATH \
        #--label_search \
        #--hold_vector \
        #--meta_model_weight_path $META_EVAL_WEIGHT_PATH \
        #--eval \
        "$@"
