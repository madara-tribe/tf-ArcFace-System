from pathlib import Path
from tqdm import tqdm
import cv2, os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from metrics import scheduler
from .cfg import Cfg

def create_gamma_img(gamma, img):
    gamma_cvt = np.zeros((256,1), dtype=np.uint8)
    for i in range(256):
        gamma_cvt[i][0] = 255*(float(i)/255)**(1.0/gamma)
    return cv2.LUT(img, gamma_cvt)


class DataLoad:
    def __init__(self, config, cosine_annealing=True):
        self.cfg = config
        self.cosine_annealing = cosine_annealing
        self.width, self.height = config.W, config.H

    def create_callbacks(self):
        checkpoint_path = os.path.join(self.cfg.WEIGHT_DIR, "arcface_model_{epoch:02d}.hdf5")
        checkpoint_dir = os.path.dirname(checkpoint_path)

        target_monitor = 'val_loss'
        cp_callback = ModelCheckpoint(checkpoint_path, monitor=target_monitor, verbose=1, save_best_only=True, mode='min')

        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1)
        
        tb_callback = TensorBoard(log_dir=self.cfg.TRAIN_TENSORBOARD_DIR,
                                      update_freq=self.cfg.train_batch * 5,
                                      profile_batch=0)
        calllbacks = [reduce_lr, cp_callback, tb_callback]
        if self.cosine_annealing:
            min_lr = 1e-3
            calllbacks.append(scheduler.CosineAnnealingScheduler(T_max=self.cfg.epochs, eta_max=self.cfg.lr, 
                           eta_min=min_lr, verbose=1))

        return calllbacks

    def preprocess(self, p, valid=None, test=None):
        if valid:
            x = cv2.imread(p)
            x = cv2.resize(x, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            if test:
                print("pass test")
                x = np.flipud(x)
            else:
                print("pass valid")
                x = create_gamma_img(1.8, x)
                #x = np.flip(x)
            x = x.reshape(self.width, self.height, 3).astype(np.float32)
        else:
            x = cv2.imread(p)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            x = x.reshape(self.width, self.height, 3).astype(np.float32)
        return x/255

    def img_load(self, valid=False, test=False):
        X, Xaug1, Xaug2, Y = [], [], [], []
        x1_dir = self.cfg.x_img
        x_imgs = os.listdir(x1_dir)
        x_imgs.sort()
        for i, image_path in enumerate(tqdm(x_imgs)):
            _, y, color, shape, _ = image_path.split("_")
            if valid:
                img = self.preprocess(os.path.join(x1_dir, image_path), valid=True, test=test)
                X.append(img)
                Y.append(int(y))
            else:
                img = self.preprocess(os.path.join(x1_dir, image_path), valid=None, test=None)
                X.append(img)
                Xaug1.append(np.flip(img))
                Xaug2.append(np.flip(img, (0, 1)))
                Y.append(int(y)) 
        X_ = X if valid else X+Xaug1+Xaug2
        Y_ = Y if valid else Y+Y+Y  
        return X_, Y_
