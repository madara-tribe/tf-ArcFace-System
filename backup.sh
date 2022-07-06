# /bin/sh
mkdir -p copy/ArcFace_model copy/meta_model
mkdir -p copy/meta_model/layers copy/meta_model/onnx copy/meta_model/results
mkdir -p copy/ArcFace_model/onnx copy/ArcFace_model/layers copy/ArcFace_model/results
mkdir -p copy/keras_efficientnet_v2
mkdir -p copy/Docker copy/metrics copy/preprocessing 

# common
cp *.py *.sh *.txt copy/
cp -r metrics/*.py copy/metrics/
cp -r Docker/* copy/Docker/
cp -r keras_efficientnet_v2/*.py copy/keras_efficientnet_v2/
cp -r preprocessing/*.py copy/preprocessing/
# arcface model
cp -r ArcFace_model/*.py ArcFace_model/*.sh ArcFace_model/*.txt copy/ArcFace_model/
cp -r ArcFace_model/layers/*.py copy/ArcFace_model/layers/
cp -r ArcFace_model/onnx/* copy/ArcFace_model/onnx/
cp -r ArcFace_model/weights copy/ArcFace_model/
cp -r ArcFace_model/logs copy/ArcFace_model/
cp -r ArcFace_model/results/* copy/ArcFace_model/results/
# meta model
cp -r meta_model/*.py meta_model/*.sh meta_model/*.txt copy/meta_model/
cp -r meta_model/layers/*.py copy/meta_model/layers/
cp -r meta_model/onnx/* copy/meta_model/onnx/
cp -r meta_model/weights copy/meta_model/
cp -r meta_model/logs copy/meta_model/
cp -r meta_model/results/* copy/meta_model/results/
