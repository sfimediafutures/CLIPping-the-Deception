from __future__ import print_function
import glob
import clip
import re
from itertools import chain
import os
import cv2
import argparse
import random
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import timm
from torchvision.transforms import ToPILImage
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import csv
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score, precision_recall_curve, f1_score
from scipy.ndimage import gaussian_filter

# Linear Probing Imports
from utils import class_labels, get_dataset, get_transforms, DeepFakeSet, get_class_based_training_data
from models import clipmodel, dinov2, CLIPModelOhja
from trainer import train_model

# Prompt Tuning Imports
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

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
import trainers.clip_zero_shot
import trainers.cocoop
import trainers.zsclip

from eval_utils import print_args, reset_cfg, extend_cfg, setup_cfg, get_parsed_args
from eval_utils_fine_tuned import print_args_fine_tuned, reset_cfg_fine_tuned, extend_cfg_fine_tuned, setup_cfg_fine_tuned, get_parsed_args_fine_tuned

def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def update_and_save_evaluation(model_name, dataset_name, accuracy, f1_score, average_precision, output_path, model_evaluations):
    # Check if the model key exists in the dictionary
    if model_name not in model_evaluations:
        model_evaluations[model_name] = {}
    # Check if the dataset key exists for the given model
    if dataset_name not in model_evaluations[model_name]:
        model_evaluations[model_name][dataset_name] = {}
    # Update evaluation results
    model_evaluations[model_name][dataset_name]["accuracy"] = accuracy
    model_evaluations[model_name][dataset_name]["f1_score"] = f1_score
    model_evaluations[model_name][dataset_name]["average_precision"] = average_precision

    output_path = output_path.replace('\\','/')
    model_name = model_name.replace('\\','/')
    
    if 'context' in model_name:
        save_file_name = output_path + model_name.split('/')[1]+'.json'
    elif 'finetuned' in model_name:
        save_file_name = output_path + model_name.split('/')[1]+'.json'
    else:
        save_file_name = output_path + '/' + model_name.split('/')[-1].split('.')[0]+'.json'
        
    # Save the updated dictionary to a JSON file
    with open(save_file_name, "w") as json_file:
        json.dump(model_evaluations, json_file, indent=2)

def eval_linear_prob(args, dataset_path, dataset_names, image_extensions, device):
    print("*************")
    print("Evaluating Linear Probing Method!")
    model_names = ['weights/CLIP_linear_prob_2_epoch_'+args.model+'_vit_large_with_augs.pth']
    model_evaluations = {}
    tfms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
    for model_name in model_names:
        model_ours = clipmodel()
        model_ours.load_state_dict(torch.load(model_name), strict=True)
        model_ours.eval()
        model_ours.cuda()
        print('==Loaded ' + model_name.split('/')[-1].split('.')[0] + '==')
        for dataset in dataset_names:
            print("*************")
            print('Evaluating on: ' + dataset)
            labels_map = ["real", "fake"]
            count = 0
            all_images = []
            predicted_labels = []
            true_labels = []
            predicted_probs = []

            dataset_path = dataset_path.replace('\\','/')
            real_image_directory = dataset_path + '/' + dataset + '/images/val/n01440764/'
            fake_image_directory = dataset_path + '/' + dataset + '/images/val/n01443537/'
            print(real_image_directory)

            images = []
            for extension in image_extensions:
                images.extend(glob.glob(os.path.join(real_image_directory, extension)))
                images.extend(glob.glob(os.path.join(fake_image_directory, extension)))
            images = sorted(images)
            images = [path.replace('\\','/') for path in images]
            print('Num. Images: ', len(images))
            y_pred = []
            for image in images:
                img = cv2.imread(image)
                # img = cv2_jpg(img, 50)
                # image = add_noise(image, 0.4)
                # gaussian_blur(img, sig)
                img = Image.fromarray(img)
                img = img.convert('RGB')
                # img = Image.open(image)
                img = tfms(img)
                y_true = []
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        outputs = model_ours(img.unsqueeze(0).to(device))
                        y_pred.extend(outputs.sigmoid().flatten().tolist())
                        # y_true.extend(label.flatten().tolist())
        
                for idx in torch.topk(outputs[0], k=1).indices.tolist():
                    prob = torch.softmax(outputs[0], 0)[idx].item()
                    if labels_map[idx] == 'real':
                        # print("real")
                        predicted_labels.append(0)
                        predicted_probs.append(1 - prob)
                    else:
                        # print("fake")
                        predicted_labels.append(1)
                        predicted_probs.append(prob)
                if 'n01440764' in image.split('/')[-2]:
                    true_labels.append(0)
                else:
                    true_labels.append(1)
            
            average_precision = 100 * average_precision_score(true_labels, predicted_probs)
            accuracy = 100 * accuracy_score(true_labels, predicted_labels)
            macro_f1 = 100.0 * f1_score(true_labels, predicted_labels, average="macro")
            
            update_and_save_evaluation(model_name, dataset, accuracy, macro_f1, average_precision, args.output, model_evaluations)
            print('--------------')
            
        torch.cuda.empty_cache()

def dummy_parse_args():
    pass

def eval_fine_tuning(args, dataset_path, dataset_names, image_extensions, device):
    print("*************")
    print("Evaluating Fine-Tuning Method!")

    if '100k' in args.model:
        model_names = ['weights/finetuned_1_epoch_100k/']
    elif '80k' in args.model:
        model_names = ['weights/finetuned_1_epoch_80k/']
    elif '60k' in args.model:
        model_names = ['weights/finetuned_1_epoch_60k/']
    elif '40k' in args.model:
        model_names = ['weights/finetuned_1_epoch_40k/']
    elif '20k' in args.model:
        model_names = ['weights/finetuned_1_epoch_20k/']
    
    model_evaluations = {}
    args.parser = dummy_parse_args()
    for dataset in dataset_names:
        coop_args = get_parsed_args_fine_tuned(model_names[0], dataset, dataset_path)
        cfg = setup_cfg_fine_tuned(coop_args)
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
        if torch.cuda.is_available() and cfg.USE_CUDA:
            print('Using CUDA!!!')
            torch.backends.cudnn.benchmark = True

        print_args(coop_args, cfg)
        print("Collecting env info ...")
        print("** System info **\n{}\n".format(collect_env_info()))

        trainer = build_trainer(cfg)
        trainer.load_model(coop_args.model_dir, epoch=coop_args.load_epoch)

        results, results_dict = trainer.test()
        update_and_save_evaluation(model_names[0], dataset, results_dict['accuracy'], results_dict['macro_f1'], results_dict['average_precision'], args.output, model_evaluations)


def eval_prompt_tuning(args, dataset_path, dataset_names, image_extensions, device):
    print("*************")
    print("Evaluating Prompt Tuning Method!")

    if '100k_16' in args.model:
        model_names = ['weights/100000_16context_best_until_now/']
    elif '100k_8' in args.model:
        model_names = ['weights/100000_8context/']
    elif '100k_4' in args.model:
        model_names = ['weights/100000_4context/']
    
    model_evaluations = {}
    splitted_string = model_names[0].split('/')[-2].split('_')[1]
    num_ctx_tokens = int(re.split('(\d+)',splitted_string)[1])
    print('Num. Context Tokens: ', num_ctx_tokens)
    args.parser = dummy_parse_args()
    for dataset in dataset_names:
        coop_args = get_parsed_args(model_names[0], dataset, num_ctx_tokens, dataset_path)
        cfg = setup_cfg(coop_args)
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
        if torch.cuda.is_available() and cfg.USE_CUDA:
            print('Using CUDA!!!')
            torch.backends.cudnn.benchmark = True

        print_args(coop_args, cfg)
        print("Collecting env info ...")
        print("** System info **\n{}\n".format(collect_env_info()))

        trainer = build_trainer(cfg)
        trainer.load_model(coop_args.model_dir, epoch=coop_args.load_epoch)

        results, results_dict = trainer.test()
        update_and_save_evaluation(model_names[0], dataset, results_dict['accuracy'], results_dict['macro_f1'], results_dict['average_precision'], args.output, model_evaluations)
        

def main(args):
    print("Starting Evaluation!")
    
    seed = 17
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.tif']
    dataset_path = args.dataset.replace("\\", "/")
    print("Dataset path: " + dataset_path)
    print("Output path: " + args.output)

    dataset_names = ['progan', 'biggan', 'cyclegan', 'eg3d', 'gaugan',  'stargan', 'stylegan', 'stylegan2', 'stylegan3', 
                 'dalle2', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'ldm_100', 'ldm_200', 'ldm_200_cfg',
                 'sd_512x512', 'sdxl', 'taming', 'deepfake', 'firefly', 'midjourney_v5', 'dalle3', 'faceswap']

    if args.variant == 'linearProbing':
        eval_linear_prob(args, dataset_path, dataset_names, image_extensions, device)
    elif args.variant == 'promptTuning':
        eval_prompt_tuning(args, dataset_path, dataset_names, image_extensions, device)
    elif args.variant == 'fineTuning':
        eval_fine_tuning(args, dataset_path, dataset_names, image_extensions, device)
    else:
        print('More methods coming soon')

    print('Evaluation completed!!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="linearProbing", choices=["linearProbing", "promptTuning", "fineTuning", "adapterNetwork"], help="name of the adaptation method")
    parser.add_argument("--model", type=str, choices=["100k", "100k_16", "100k_8", "100k_4"], default="100k", help="name of linear probing model to evaluate")
    parser.add_argument("--dataset", type=str, default="", help="path to dataset")
    parser.add_argument("--output", type=str, default="", help="output directory to write results")
    
    args = parser.parse_args()
    main(args)