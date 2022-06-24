#!/bin/sh
ARC_WIGHT_PATH="ArcFace_model/weights/122arcface_model_29.hdf5"
python3 main.py \
        --meta_model_train
        #--arcface_model_eval --arcface_model_weight_path $ARC_WIGHT_PATH
        #--arcface_model_train
        #--meta_model_train \
