#!/bin/sh
ARC_WIGHT_PATH="ArcFace_model/weights/ep29_122arcface_model.hdf5"
META_WEIGHT_PATH='meta_model/weights/ep27_meta_model.hdf5'
python3 main.py \
        --arcface_model_eval --arcface_model_weight_path $ARC_WIGHT_PATH
        #--meta_model_eval --meta_model_weight_path $META_WEIGHT_PATH
        #--arcface_model_train
        #--meta_model_train \
