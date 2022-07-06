import os, sys
sys.path.append('../cfg')
os.environ['TF_KERAS'] = '1'
from keras2onnx import convert_keras
import onnx
from onnx_test import MetaTester
from cfg import Cfg as csCfg

OUTPUT_ONNX_MODEL_NAME = 'meta_model.onnx'
cscfg = csCfg

def main(weight_path):
    meta_model = MetaTester(cscfg).load_model(weight_path)
    print(meta_model.name)
    
    onnx_model = convert_keras(meta_model, meta_model.name)
    onnx.save(onnx_model, OUTPUT_ONNX_MODEL_NAME)
    print("success to output as "+OUTPUT_ONNX_MODEL_NAME)

if __name__=="__main__":
    weight_path = '../results/ep66_260x260_meta_model.hdf5'
    main(weight_path)


