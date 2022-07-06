import os, sys
sys.path.append('../')
os.environ['TF_KERAS'] = '1'
from tensorflow.keras.models import Model
from keras2onnx import convert_keras
import onnx
from load_model_ import load_arcface_model as load_pretrain_model


OUTPUT_ONNX_MODEL_NAME = 'arcface_model.onnx'
N = -3
def load_arcface_model(model):
    # arcface model
    embed_inputs = model.get_layer(index=0).input
    embed_out = model.get_layer(index=N).output
    arcface_model = Model(inputs=embed_inputs, outputs=embed_out)
    arcface_model.summary()
    return arcface_model

def main(weight_path):
    model = load_pretrain_model(weights=weight_path)
    arcface_model = load_arcface_model(model)
    print(arcface_model.name)
    
    onnx_model = convert_keras(arcface_model, arcface_model.name)
    onnx.save(onnx_model, OUTPUT_ONNX_MODEL_NAME)
    print("success to output as "+OUTPUT_ONNX_MODEL_NAME)

if __name__=='__main__':
    weight_path = '../results/ep32_260x260_arcface_model.hdf5'
    main(weight_path)

