from pathlib import Path
from tqdm import tqdm
import cv2, os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from metrics import scheduler, padding_resize

EPOCHS = 40
BATCH_SIZE = 4
HEIGHT = WIDTH = 260
WEIGHT_DIR = 'weights'
LR = 1e-1
RACE_NUM_CLS = 5
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}

class UTKLoad:
    def __init__(self, gamma=2.0, cosine_annealing=True):
        self.gamma_ = gamma
        self.cosine_annealing = cosine_annealing
        
    def create_callbacks(self):
        checkpoint_path = os.path.join(WEIGHT_DIR, "arcface_model_{epoch:02d}.hdf5")
        checkpoint_dir = os.path.dirname(checkpoint_path)

        target_monitor = 'val_loss'
        cp_callback = ModelCheckpoint(checkpoint_path, monitor=target_monitor, verbose=1, save_best_only=True, mode='min')

        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1)
        
        tb_callback = TensorBoard(log_dir='logs/',
                                      update_freq=BATCH_SIZE * 5,
                                      profile_batch=0)
        calllbacks = [reduce_lr, cp_callback, tb_callback]
        if self.cosine_annealing:
            min_lr = 1e-3
            calllbacks.append(scheduler.CosineAnnealingScheduler(T_max=EPOCHS, eta_max=LR, eta_min=min_lr, verbose=1))

        return calllbacks


    def gamma(self, img, gamma = 0.7):
        gamma_cvt = np.zeros((256,1),dtype = 'uint8')
        for i in range(256):
             gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
        gamma_img = cv2.LUT(img, gamma_cvt)
        return gamma_img

    def load_data(self, path="UTK", img_size=224):
        image_dir = Path(path)
        out_imgs, races = [], []
        for i, image_path in enumerate(tqdm(image_dir.glob("*.jpg"))):
            image_name = image_path.name  # [age]_[gender]_[race]_[date&time].jpg
            age, gender, race, _ = image_name.split("_")
            races.append(int(race))
            img = cv2.imread(str(image_path))
            #img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            img = padding_resize.img_padding(img, desired_size=img_size)
            #img = self.gamma(img, gamma = self.gamma_)
            out_imgs.append(img)
            if i==4000:
                break
        return out_imgs, races


