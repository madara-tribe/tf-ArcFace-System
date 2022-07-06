#!/bin/sh
ARC_WIGHT_PATH="ArcFace_model/results/ep32_260x260_arcface_model.hdf5"
META_EVAL_WEIGHT_PATH='meta_model/results/ep66_260x260_meta_model.hdf5'
META_TRAIN_WEIGHT_PATH='meta_model/hdf5/ep25_meta2.hdf5'
python3 main.py \
        --main \
        --predict \
        --arcface_model_weight_path $ARC_WIGHT_PATH \
        --meta_model_weight_path $META_EVAL_WEIGHT_PATH \
        #--meta \
        #--arcface \
        "$@"
