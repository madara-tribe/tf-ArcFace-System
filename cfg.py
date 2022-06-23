import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.W = 256
Cfg.H = 256
Cfg.channels = 3
Cfg.train_batch = 4
Cfg.val_batch = 1
Cfg.lr = 0.001
Cfg.epochs = 30
Cfg.val_interval = 400
Cfg.gpu_id = '3'
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
Cfg.classes=122
Cfg.start_fm = 64
Cfg.embed_size = 512 # CNN Encoder outut size

## dataset
Cfg.x_img = "data/x_img"
Cfg.save_checkpoint = True
Cfg.WEIGHT_DIR = "./weights"
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')
