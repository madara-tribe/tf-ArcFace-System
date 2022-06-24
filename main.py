import os, sys
import time
import argparse
from datetime import datetime
import numpy as np
from meta_model.cfg import Cfg as csCfg
from meta_model.train import MetaTrainer
from ArcFace_model.cfg import Cfg
from ArcFace_model.train import Trainer
from ArcFace_model.test import Tester

def main():
    EXECDATETIME = datetime.now().strftime('%Y%m%d-%H%M')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--arcface_model_weight_path", default=None,
                            help="arcface_model weight path")
    parser.add_argument("--arcface_model_train", action="store_true",
                            help="arcface_model train")
    parser.add_argument("--arcface_model_eval", action="store_true",
                            help="arcface_model eval")

    parser.add_argument("--meta_model_weight_path", default=None,
                            help="meta_model weight path")
    parser.add_argument("--meta_model_train", action="store_true",
                            help="meta_model train")
    parser.add_argument("--meta_model_eval", action="store_true",
                            help="meta_model eval")
    args = parser.parse_args()

    cscfg = csCfg
    if args.meta_model_weight_path:
        weight_path = args.meta_modeel_weight_path
    else:
        weight_path = None
        os.makedirs(cscfg.WEIGHT_DIR, exist_ok=True)

    if args.meta_model_train:
        MetaTrainer(cscfg).train(weight_path=weight_path)
    
    cfg = Cfg
    if args.arcface_model_weight_path:
        weight_path = args.arcface_model_weight_path
    else:
        weight_path = None
        os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
 
    if args.arcface_model_train:
        Trainer(cfg).train(weight_path=weight_path)
    elif args.arcface_model_eval:
        weight_path = args.arcface_model_weight_path
        Tester(cfg).test(weight_path)
if __name__ == '__main__':
    main()
