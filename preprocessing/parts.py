import cv2
import numpy as np
import json

def square_resize(img):
    max_length = max(img.shape)
    
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(max_length)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = max_length - new_size[1]
    delta_h = max_length - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    ret_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    return ret_img

def crop_resize(img_, n, resize):
    img = square_resize(img_)
    img = cropCore(img, n, resize)
    return img


def cropCore(imgs, n, resize):
    h, w = imgs.shape[:2]
    hratio = int((h/2)/n)
    wratio = int((w/2)/n)
    img = imgs[wratio:-wratio, hratio:-hratio]
    return cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)

def load_json(p="train_meta.json"):
    with open(p, 'r') as j:
        data = json.loads(j.read())
        #pprint.pprint(data)
    return data


