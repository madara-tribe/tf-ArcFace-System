import os, cv2
import glob, time
from pathlib import Path
import numpy as np
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from UTKload import UTKLoad, WEIGHT_DIR, WIDTH, HEIGHT, RACE_NUM_CLS
from train import ArcFace
from metrics.cosin_metric import cosine_similarity


ms = MinMaxScaler()#.fit_transform(X)
ss = StandardScaler() #.fit_transform(X)

def load_embeding_model(weight):
    arcface_ = ArcFace(train_path=None, val_path=None, num_race=RACE_NUM_CLS)
    m = arcface_.load_arcface_model(weight)
    embeding_model = Model(m.get_layer(index=0).input, m.get_layer(index=-3).output)
    embeding_model.summary()
    return embeding_model
    
def load_querys(model, test_path):
    load_ = UTKLoad(gamma=2.0)
    X_test, y_test = load_.load_data(path=test_path, img_size=HEIGHT)
    X_test = np.array(X_test, dtype='float32')/255
    Xs = model.predict(X_test, verbose=1)
    return Xs, y_test
  
def get_hold_vector(model, path):
    image_dir = Path(path)
    X_test, vector_label = [], []
    for i, image_path in enumerate(image_dir.glob("*.jpg")):
        image_name = image_path.name
        y_label = image_name.split("_")[0]
        img = cv2.imread(str(image_path))
        vector_label.append(y_label)
        X_test.append(img)
    #load_ = UTKLoad(gamma=2.0)
    #hold_vector, vector_label = load_.load_data(path=path, img_size=HEIGHT)
    hold_vector = np.array(X_test, dtype='float32')/255
    hold_vector = model.predict(hold_vector, verbose=1)
    return hold_vector, vector_label
    
    

def test_performance(model, query_path, vector_path):
    X_querys, y_test = load_querys(model, query_path)
    #X_querys = ss.fit_transform(X_querys)
    X_querys /=np.linalg.norm(X_querys, axis=1, keepdims=True)
    
    hold_vector, vector_label = get_hold_vector(model, vector_path)
    #hold_vector = ss.fit_transform(hold_vector)
    hold_vector /=np.linalg.norm(hold_vector, axis=1, keepdims=True)
    
    acc = 0
    start = time.time()
    for i, (xq, y_label) in enumerate(zip(X_querys, y_test)):
        #highest_idx = np.argmax(cosine_similarity(xq, hold_vector))
        similarity = cosine_similarity(xq, hold_vector)
        print(similarity[0])
        label_candidate = [vector_label[i] for i, sim in enumerate(similarity[0])]
        frequent_idx = np.argmax(np.bincount(label_candidate))
        if y_label==frequent_idx:
            print('right !!')
            acc += 1
        else:
            print('wrong !!')
            acc += 0
    print("TF Model Inference Latency is", time.time() - start, "[s]")
    print('accuracy is {}'.format(acc/len(X_querys)))
        


if __name__=='__main__':
    weight = 'arcface_model_10.hdf5'
    query_path = '../../UTK/UTKFace'
    vector_path = 'H'
    embeding_model = load_embeding_model(weight)
    test_performance(embeding_model, query_path, vector_path)


