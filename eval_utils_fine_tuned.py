import argparse
import torch
import glob

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom

import datasets.imagenet
import datasets.guided
import datasets.biggan
import datasets.cyclegan
import datasets.dalle2
import datasets.deepfake
import datasets.gaugan
import datasets.glide_50_27
import datasets.glide_100_10
import datasets.glide_100_27
import datasets.ldm_100
import datasets.ldm_200
import datasets.ldm_200_cfg
import datasets.stargan
import datasets.stylegan
import datasets.stylegan2
import datasets.stylegan3
import datasets.sd_512x512
import datasets.sdxl
import datasets.dalle3
import datasets.taming
import datasets.eg3d
import datasets.firefly
import datasets.midjourney_v5
import datasets.progan
import datasets.faceswap

import trainers.coop
import trainers.clip_adapter
import trainers.cocoop
import trainers.zsclip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    # if args.num_ctx_tokens:
    #     cfg.TRAINER.COOP.N_CTX = args.num_ctx_tokens


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "front"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new



def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg

def get_parsed_args(model_dir, dataset_name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../Datasets/ICMRDataset/test/deepfake_eval/", help="path to dataset")
    # parser.add_argument("--deepfake-set", default="biggan", action="store_true")        
    parser.add_argument("--output-dir", type=str, default="../CoOp/outputs/", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=17, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="../CoOp/configs/trainers/coop/vit_l14_ep2.yaml", help="path to config file"
    )
    parser.add_argument("--dataset-config-file", type=str, default="../CoOp/configs/datasets/"+str(dataset_name)+".yaml",
        help="path to config file for dataset setup",)
    parser.add_argument("--trainer", type=str, default="CLIP_Adapter", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", default="True", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=model_dir,
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, default="1", help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    # parser.add_argument(
    #     "--num_ctx_tokens", default=num_ctx_tokens, help="do not call trainer.train()"
    # )
    
    args = parser.parse_args()

    return args