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
To classify face to label 0~4 (0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others') by [UTKFace](https://susanqq.github.io/UTKFace/) Dataset.

This time, the model is EfficientNet-V2 with ArcFace.

<img src="https://user-images.githubusercontent.com/48679574/147999774-478d64d8-0961-499e-bb8e-a6d7740acf32.png" width="600px">


# Result

## ArcFace perfomance

Best accuracy is ArcFace. <b>(ArcFace > SphereFace > CosFace)</b>

train dataset is 20037, validation dataset is 3000

accuracy is <b>95%</b> to classify 0~4(0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others').

When prediction, to search similar image from other 50 images

<b>When ArcFace prediction from hold_vector</b>
```python
# get most similar top N images from hold vector
similarity = cosine_similarity(xq, hold_vector)
label_candidate = [vector_label[idx] for idx in np.argsort(similarity[0])[::-1][:20]]
frequent_idx = np.argmax(np.bincount(label_candidate))


>>>>
# 1000 images prediction time is
TF Model Inference Latency is 0.40515995025634766 [s]
```


## Data-Centric approrch
- optimizer Adam
- category cross entropy
- image padding resize

model customize is not so much. Instead of model customization, to improve accuracy approrch is mainly data preorocessing and data cleaning(ata-Centric approrch)

<img src="https://user-images.githubusercontent.com/48679574/147999782-4e9e84cc-09f1-4a15-994b-1a2cb1f8e8b1.jpeg" width="500px">

[A data-centric approach to understanding the pricing of financial options](https://www.researchgate.net/publication/225829199_A_data-centric_approach_to_understanding_the_pricing_of_financial_options)

# ONNX Convert

## tensorflow model size
```
Total params: 8,769,374
Trainable params: 8,687,086
Non-trainable params: 82,288
```

## ONNX Latency 
```The ONNX operator number change on the optimization: 927 -> 481```

<b>latency</b> is 
```
right
ONNX Inference Latency is 192.74330139160156 [ms]
```


# Note (useful tecs)


# References

- [keras_efficientnet_v2](https://github.com/leondgarse/keras_efficientnet_v2/blob/main/keras_efficientnet_v2/progressive_train_test.py)
- [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py)

