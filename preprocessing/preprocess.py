import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json 
import pprint
import glob
from parts import crop_resize, load_json

H = W = resize = 224
colorlist = ['YW', 'A3', 'DG', 'YS', 'CC', 'YB', 'CH', 'YK', 'WM', 'E5', 'RB']
colorlist2 = ['white', 'calmgray', 'darkgray', 'silver', 'antiquegray', 'bronze', 'Stainless', 'black', 
             'silkywhite', 'gray', 'randomnumber']
save_path = "wav"
data = load_json()

def create_img_for_arcface(path='train/'):
    save_path="wav/color"
    count = 0
    label = None
    for i in range(121+1):
        if i < 10:
            i = "00"+str(i)
        elif i<100 and i>9:
            i = "0"+str(i)
        elif i>99:
            i = str(i)
        img_ = path + i + "/0.jpg"
        imgs = cv2.imread(img_)
        imgs = crop_resize(imgs, n=8, resize=resize)
        #imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY).reshape(H, W, 1)

        if data[i]["category"]=="戸車":
            shape_label = 0
        elif data[i]["category"]=="クレセント":
            shape_label = 1
        if data[i]["color"] in colorlist:
            color_idx = colorlist.index(str(data[i]["color"]))
        idx = count   
        label, color_idx, shape_label = str(int(i)), str(color_idx), str(shape_label)
        print(label, imgs.max(), imgs.min())
        if count < 10:
            idx = "000"+str(idx)
        elif count<100 and count>9:
            idx = "00"+str(idx)
        elif count<1000 and count>99:
            idx = "0"+str(idx)
        elif count>999:
            idx = str(idx)
        cv2.imwrite(os.path.join(save_path, str(i) +"_"+ label +"_" +color_idx+ "_" +shape_label + "_.jpg"), imgs)
        count += 1
        if int(count)<10:
            plt.imshow(imgs, "gray"),plt.show()
            
            
def preprocess_and_save():
    path="train/"
    count = 0
    for i in range(121+1):
        if i < 10:
            i = "00"+str(i)
        elif i<100 and i>9:
            i = "0"+str(i)
        elif i>99:
            i = str(i)
        imglist = glob.glob(path + i + "/*.jpg")
        label = None
        for idx, img_ in enumerate(imglist):
            imgs = cv2.imread(img_)
            imgs = crop_resize(imgs, n=8, resize=resize)
            #imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY).reshape(H, W, 1)
            if data[i]["category"]=="戸車":
                shape_label = 0
            elif data[i]["category"]=="クレセント":
                shape_label = 1
            if data[i]["color"] in colorlist:
                color_idx = colorlist.index(str(data[i]["color"]))

            label, color_idx, shape_label = str(int(i)), str(color_idx), str(shape_label)
            print(count, label, imgs.shape, imgs.max(), imgs.min())
            idx = count
            if count < 10:
                idx = "000"+str(idx)
            elif count<100 and count>9:
                idx = "00"+str(idx)
            elif count<1000 and count>99:
                idx = "0"+str(idx)
            elif count>999:
                idx = str(idx)
            cv2.imwrite(os.path.join(save_path, idx + "_" + label +"_"+ color_idx+"_" +shape_label+ "_.jpg"), imgs)
            count += 1
            if int(count)<10:
                plt.imshow(imgs, "gray"),plt.show()

if __name__=="__main__":
    #create_img_for_arcface()
    preprocess_and_save()




