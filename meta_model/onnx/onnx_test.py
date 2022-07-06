import os, sys
sys.path.append("../")
import numpy as np
import os, sys
from tensorflow.keras.models import Model
from layers.resnet import create_model

class MetaTester:
    def __init__(self, config):
        self.cfg = config
    
    def load_model(self, weights): 
        SHAPE_OUTPUT_IDX = -9
        COLOR_OUTPUT_IDX = -10
        model = create_model(self.cfg.H, self.cfg.W, k=1, lr=1e-3)
        model.load_weights(weights)
        
        inputs = model.get_layer(index=0).input
        shape_logits = model.get_layer(index=SHAPE_OUTPUT_IDX).output
        color_logits = model.get_layer(index=COLOR_OUTPUT_IDX).output
        meta_model = Model(inputs=inputs, outputs=[shape_logits, color_logits])
        meta_model.summary()
        return meta_model
