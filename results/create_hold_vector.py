from pathlib import Path
from tqdm import tqdm
import cv2, os
import numpy as np
from UTKload import HEIGHT, WIDTH
from metrics import padding_resize


HEIGHT = WIDTH = 260
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
R = 'hold_vector'

def load_data(path="UTK", img_size=224):
    image_dir = Path(path)
    out_imgs, races = [], []
    for i, image_path in enumerate(tqdm(image_dir.glob("*.jpg"))):
        image_name = image_path.name  # [age]_[gender]_[race]_[date&time].jpg
        age, gender, race, name = image_name.split("_")
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        #img = padding_resize.img_padding(img, desired_size=img_size)
        #img = self.gamma(img, gamma = self.gamma_)
        #out_imgs.append(img)
        if int(race)==0:
            cv2.imwrite(os.path.join(R, ID_RACE_MAP[0], str(race)+'_'+str(name)), img)
        elif int(race)==1:
            cv2.imwrite(os.path.join(R, ID_RACE_MAP[1], str(race)+'_'+str(name)), img)
        elif int(race)==2:
            cv2.imwrite(os.path.join(R, ID_RACE_MAP[2], str(race)+'_'+str(name)), img)
        elif int(race)==3:
            cv2.imwrite(os.path.join(R, ID_RACE_MAP[3], str(race)+'_'+str(name)), img)
        elif int(race)==4:
            cv2.imwrite(os.path.join(R, ID_RACE_MAP[4], str(race)+'_'+str(name)), img)
        if i==300:
            break

if __name__=='__main__':
    path = '../../UTK/UTKFace'
    os.makedirs(os.path.join(R, ID_RACE_MAP[0]), exist_ok=True)
    os.makedirs(os.path.join(R, ID_RACE_MAP[1]), exist_ok=True)
    os.makedirs(os.path.join(R, ID_RACE_MAP[2]), exist_ok=True)
    os.makedirs(os.path.join(R, ID_RACE_MAP[3]), exist_ok=True)
    os.makedirs(os.path.join(R, ID_RACE_MAP[4]), exist_ok=True)
    load_data(path=path, img_size=HEIGHT)

