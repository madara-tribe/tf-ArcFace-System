import json
import numpy as np
    
def arcface_test():
    shape_cls = 2
    color_cls = 11
    dd = json.load(open("data/cs_label.json"))
    pred_color = np.load("data/npy/color_sim600.npy")
    pred_shape = np.load("data/npy/shape_sim600.npy")
    cossims = np.load("data/npy/cossim600.npy")
    confs = (pred_shape/shape_cls)+(pred_color/color_cls)
    #sl, cl = v['category'], v['color']
    embbed = [conf+val for conf, val in zip(confs, cossims)]
    print(np.argmax(embbed), np.argmax(cossims), np.argmax(confs))
    
if __name__=="__main__":
    arcface_test()
