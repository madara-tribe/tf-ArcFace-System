# Versions
```
- Python==3.7.0
- tensorflow==2.3.0
- Keras==2.3.1
- onnx==1.9.0
- onnxruntime==1.8.1
- keras2onnx==1.9.0
- cuda 10.1
```

# Abstract

with 122 labels and shape and color lanel (meta label), using Arcface model and meta-model for image classification task of complicated and small amount of data.



# Performance

| Model | Head | class | accuracy |
| :---         |     :---:      |     :---:      |         ---: |
| ArcFace model| Dense+sodtmax   | 122 | 76%|
| meta model | ArcFace+softmax   | 11  | 82%|
| ArcFace model + meta model | similar-image-search   | 122  | 96%|


## Arcface model

## meta model





## Data-Centric approrch
- optimizer Adam
- category cross entropy
- image padding resize

model customize is not so much. Instead of model customization, to improve accuracy approrch is mainly data preorocessing and data cleaning(data-Centric approrch)

<b>image padding resize(example)</b>

<img src="https://user-images.githubusercontent.com/48679574/147999782-4e9e84cc-09f1-4a15-994b-1a2cb1f8e8b1.jpeg" width="500px">



# References
- [keras_efficientnet_v2](https://github.com/leondgarse/keras_efficientnet_v2/blob/main/keras_efficientnet_v2/progressive_train_test.py)
- [A data-centric approach to understanding the pricing of financial options](https://www.researchgate.net/publication/225829199_A_data-centric_approach_to_understanding_the_pricing_of_financial_options)

