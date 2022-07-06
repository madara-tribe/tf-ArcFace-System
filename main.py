import os
import time
import argparse
from meta_model.cfg import Cfg as csCfg
from meta_model.train import MetaTrainer
from meta_model.test import MetaTester

from ArcFace_model.cfg import Cfg
from ArcFace_model.train import Trainer
from ArcFace_model.test import Tester

from predict import Predictor

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--main", action="store_true",
                            help="to work main.sh and predict.py")
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
    parser.add_argument("--predict", action="store_true",
                            help="predict final label with arcface and meta model")
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
    elif args.arcface:
        cfg = Cfg
        if args.train:
            weight_path = args.arcface_model_weight_path
            os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
            Trainer(cfg).train(weight_path=weight_path)   
        elif args.eval:
            weight_path = args.arcface_model_weight_path
            Tester(cfg).test(weight_path)
    elif args.main:
        if args.predict:
            candidates = 20
            from ArcFace_model.load_model import load_arcface_model as load_pretrain_model
            cscfg = csCfg
            cfg = Cfg
            meta_weight_path = args.meta_model_weight_path
            meta_model = MetaTester(cscfg).load_model(meta_weight_path)
            arcface_weight_path = args.arcface_model_weight_path
            pretrained_model = load_pretrain_model(weights=arcface_weight_path)
            arcface_model = Tester(cfg).load_arcface_model(pretrained_model)
            Predictor(cscfg, meta_model, arcface_model).predict(num_candidates=candidates)

if __name__ == '__main__':
    main()

