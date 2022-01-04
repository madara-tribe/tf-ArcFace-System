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




# Result

## ArcFace perfomance

Best accuracy is ArcFace. <b>(ArcFace > SphereFace > CosFace)</b>

train dataset is 20037, validation dataset is 3000

accuracy is <b>95%</b> to classify 0~4(0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others').

When prediction, to search similar image from other 50 images

<b>When ArcFace prediction from hold_vector</b>
```python
# test.py
similarity = cosine_similarity(xq, hold_vector)
label_candidate = [vector_label[idx] for idx in np.argsort(similarity[0])[::-1][:20]]
frequent_idx = np.argmax(np.bincount(label_candidate))


>>>>
# 1000 images prediction time is
TF Model Inference Latency is 0.40515995025634766 [s]
```


## Kmeans++ plot
After training, prediction result plot by k-means++



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
- optimizer SGD
- category cross entropy
- image padding resize

<img src="https://user-images.githubusercontent.com/48679574/145761032-264e07fc-a5c5-4048-87ce-41c52dc97a74.png" width="400px">


# References

- [keras_efficientnet_v2](https://github.com/leondgarse/keras_efficientnet_v2/blob/main/keras_efficientnet_v2/progressive_train_test.py)
- [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py)

