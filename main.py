import os, sys
import time
import argparse
from datetime import datetime
import numpy as np
from meta_model.cfg import Cfg as csCfg
from meta_model.train import MetaTrainer
from meta_model.test import MetaTester
from meta_model.search import LabelingSearch

from ArcFace_model.cfg import Cfg
from ArcFace_model.train import Trainer
from ArcFace_model.test import Tester

def main():
    EXECDATETIME = datetime.now().strftime('%Y%m%d-%H%M')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--main", action="store_true",
                            help="to work main.sh")
    parser.add_argument("--meta", action="store_true",
                            help="to work meta in main.sh")
    parser.add_argument("--arcface", action="store_true",
                            help="to work ArcFace in main.sh")
    parser.add_argument("--arcface_model_weight_path", default=None,
                            help="arcface_model weight path")
    parser.add_argument("--meta_model_weight_path", default=None,
                            help="meta_model weight path")
    parser.add_argument("--train", action="store_true",
                            help="model train")
    parser.add_argument("--eval", action="store_true",
                            help="meta_model eval")
    parser.add_argument("--label_search", action="store_true",
                            help="easy and hard label seach for meta_model")
    parser.add_argument("--hold_vector", action="store_true",
                            help="create hold vector for each model")
    args = parser.parse_args()

    if args.meta:
        cscfg = csCfg
        if args.train:
            if args.meta_model_weight_path:
                weight_path = args.meta_model_weight_path
            else:
                weight_path = None
            os.makedirs(cscfg.WEIGHT_DIR, exist_ok=True)
            MetaTrainer(cscfg).train(weight_path=weight_path)
        elif args.eval: 
            weight_path = args.meta_model_weight_path
            MetaTester(cscfg).test(weight_path)
        elif args.label_search:
            weight_path = args.meta_model_weight_path
            LabelingSearch(cscfg).test(weight_path)       
        elif args.hold_vector:
            weight_path = args.meta_model_weight_path
            MetaTester(cscfg).create_holod_vector(weight_path) 
    elif args.arcface:
        cfg = Cfg
        if args.train:
            weight_path = args.arcface_model_weight_path
            os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
            Trainer(cfg).train(weight_path=weight_path)   
        elif args.eval:
            weight_path = args.arcface_model_weight_path
            Tester(cfg).test(weight_path)


if __name__ == '__main__':
    main()
