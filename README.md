# Versions
```
- Python==3.7.0
- tensorflow==2.3.0
- onnx==1.11.0
- keras2onnx==1.9.0
```

# Abstract

with 122 labels and shape and color lanel (meta label), using Arcface model and meta-model for image classification task of complicated and small amount of data.

<img src="https://user-images.githubusercontent.com/48679574/177512038-5c6a147d-94e7-4c6e-abda-8724c79df2da.png" width="400px">


# Performance

| Model | Head | class | accuracy |
| :---         |     :---:      |     :---:      |         ---: |
| ArcFace-model| Dense+softmax   | 122 | 76%|
| meta-model | ArcFace+softmax   | 11  | 82%|
| ArcFace-model + meta-model | similar-image-search   | 122  | 96%|


## Arcface model

<img src="https://user-images.githubusercontent.com/48679574/177523259-2c21ad54-10f5-4d21-ac05-fdc9cf3fecf5.png" width="300px"><img src="https://user-images.githubusercontent.com/48679574/177523273-a55558f3-c397-4508-a687-6ff8510d6b3f.png" width="300px">

## meta model

<img src="https://user-images.githubusercontent.com/48679574/177523322-50cad032-20f5-4548-ac65-d68c3e109d3d.png" width="300px"><img src="https://user-images.githubusercontent.com/48679574/177523335-4c689fe3-46e2-4e74-826f-6b095ec148f4.png" width="300px">


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

