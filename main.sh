#!/bin/sh
ARC_WIGHT_PATH="ArcFace_model/weights/ep21_ArcFace.hdf5"
META_EVAL_WEIGHT_PATH='meta_model/weights/arcface_model_25.hdf5'
META_TRAIN_WEIGHT_PATH='meta_model/ep28_meta.hdf5'
python3 main.py \
        --meta_model_eval --meta_model_weight_path $META_EVAL_WEIGHT_PATH 
   #--meta_model_train --meta_model_weight_path $META_TRAIN_WEIGHT_PATH
   #     --meta_model_eval --meta_model_weight_path $META_EVAL_WEIGHT_PATH        
#--arcface_model_train
        #--arcface_model_eval --arcface_model_weight_path $ARC_WIGHT_PATH
        #--arcface_model_train
        #--meta_model_train \
        "$@"
