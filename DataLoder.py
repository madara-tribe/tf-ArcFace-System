from pathlib import Path
from tqdm import tqdm
import cv2, os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from metrics import scheduler, padding_resize

WEIGHT_DIR = 'weights'

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

    def preprocess(self, p, clannel):
        if clannel==3:
            x = cv2.imread(p)
            x = cv2.resize(x, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            x = x.reshape(self.width, self.height, 3).astype(np.float32)
            #x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        elif clannel==1:
            x = cv2.imread(p, 0)
            x = cv2.resize(x, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            x = x.reshape(self.width, self.height, 1).astype(np.float32)
        return x/255

    def load_data(self, valid=False):
        x1_dir = self.cfg.x_img
        x_imgs = os.listdir(x1_dir)
        x_imgs.sort()
        
        color_meta, shape_meta= [], []
        x_label, c_label, s_label = np.load(self.cfg.x_label), np.load(self.cfg.y1), np.load(self.cfg.y2)
        X, x_colors, x_shapes, y_labels, color_label, shape_label = [], [], [], [], [], []
        for _ in range(len(x_imgs)):
            color_meta.append(np.load(self.cfg.x2))
        for _ in range(len(x_imgs)):
            shape_meta.append(np.load(self.cfg.x3))
        for i, image_path in enumerate(tqdm(x_imgs)):
            img = self.preprocess(os.path.join(x1_dir, image_path), 3)
            if valid:
                img = np.flip(img)
            # x_img
            X.append(img)
            # x2, x3
            x_colors.append(color_meta[i])
            x_shapes.append(shape_meta[i])
            # x_label, y1, y2
            y_labels.append(x_label[i])
            color_label.append(c_label[i])
            shape_label.append(s_label[i])
        return X, x_colors, x_shapes, y_labels, color_label, shape_label

