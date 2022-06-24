# /bin/sh
mkdir -p copy/ArcFace_model copy/meta_model
mkdir -p copy/meta_model/layers copy/meta_model/onnx 
mkdir -p copy/ArcFace_model/onnx copy/ArcFace_model/layers
mkdir -p copy/keras_efficientnet_v2
mkdir -p copy/Docker copy/metrics

# common
cp *.py *.sh *.txt copy/
cp -r metrics/*.py copy/metrics/
cp -r Docker/* copy/Docker/
cp -r keras_efficientnet_v2/*.py copy/keras_efficientnet_v2/

# arcface model
cp -r ArcFace_model/*.py ArcFace_model/*.sh ArcFace_model/*.txt copy/ArcFace_model/
cp -r ArcFace_model/layers/*.py copy/ArcFace_model/layers/
cp -r ArcFace_model/onnx/*.py copy/ArcFace_model/onnx/
cp -r ArcFace_model/weights copy/ArcFace_model/
cp -r ArcFace_model/logs copy/ArcFace_model/
# meta model
cp -r meta_model/*.py meta_model/*.sh meta_model/*.txt copy/meta_model/
cp -r meta_model/layers/*.py copy/meta_model/layers/
cp -r meta_model/onnx/*.py copy/meta_model/onnx/
