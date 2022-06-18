from pathlib import Path
from tqdm import tqdm
import cv2, os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from metrics import scheduler, padding_resize

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
        
        tb_callback = TensorBoard(log_dir='logs/',
                                      update_freq=self.cfg.train_batch * 5,
                                      profile_batch=0)
        calllbacks = [reduce_lr, cp_callback, tb_callback]
        if self.cosine_annealing:
            min_lr = 1e-3
            calllbacks.append(scheduler.CosineAnnealingScheduler(T_max=self.cfg.epochs, eta_max=self.cfg.lr, 
                           eta_min=min_lr, verbose=1))

        return calllbacks

    def preprocess(self, p, clannel, valid=None):
        if clannel==3:
            x = cv2.imread(p)
            x = cv2.resize(x, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            x = x.reshape(self.width, self.height, 3).astype(np.float32)
            #x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        elif clannel==1:
            x = cv2.imread(p, 0)
            x = cv2.resize(x, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            x = x.reshape(self.width, self.height, 1).astype(np.float32)
        elif valid:
            x = cv2.imread(p)
            x = cv2.resize(x, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            x = create_gamma_img(1.8, x)
            x = x.reshape(self.width, self.height, 3).astype(np.float32)
        return x/255

    def img_load(self, valid=False):
        X, y_labels, X_aug = [], [], []
        x1_dir = self.cfg.x_img
        x_imgs = os.listdir(x1_dir)
        x_imgs.sort()
    
        x_label = np.load(self.cfg.x_label)
        for i, image_path in enumerate(tqdm(x_imgs)):
            if valid:
                img = self.preprocess(os.path.join(x1_dir, image_path), 3, valid=True)
                X.append(img)
                y_labels.append(x_label[i])
            else:
                img = self.preprocess(os.path.join(x1_dir, image_path), 3, valid=None)
                aug_img = np.flip(img)
                X.append(img)
                y_labels.append(x_label[i])
                X_aug.append(aug_img)
        return X, y_labels, X_aug

    def meta_load(self, valid=False):
        x1_dir = self.cfg.x_img
        x_imgs = os.listdir(x1_dir)
        x_imgs.sort()
        
        c_label, s_label = np.load(self.cfg.y1), np.load(self.cfg.y2)
        X, X_aug, color_label, shape_label = [], [], [], []
                                    
        for i, image_path in enumerate(tqdm(x_imgs)):
            if valid:
                img = self.preprocess(os.path.join(x1_dir, image_path), 3)
            else:
                img = self.preprocess(os.path.join(x1_dir, image_path), 3)
                aug_img = np.flip(img)
                X_aug.append(aug_img)
            # img
            X.append(img)
            # x_label, y1, y2
            color_label.append(c_label[i])
            shape_label.append(s_label[i])
        return X, X_aug, color_label, shape_label


