# /bin/sh
mkdir -p copy/Docker copy/metrics copy/onnx
mkdir -p copy/keras_efficientnet_v2
cp -r *.py *.sh copy/
cp -r keras_efficientnet_v2/*.py copy/keras_efficientnet_v2/
cp -r Docker/* copy/Docker/
cp -r metrics/*.py copy/metrics/
cp -r onnx/*.py copy/onnx/
