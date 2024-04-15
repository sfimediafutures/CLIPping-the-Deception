from __future__ import print_function
import glob
import clip
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

# Local imports
from utils import class_labels, get_dataset, get_transforms, DeepFakeSet, get_class_based_training_data
from models import clipmodel, dinov2, CLIPModelOhja
from trainer import train_model

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

def main(args):
    print("Starting Evaluation!")
    
    seed = 17
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.tif']  # Add more extensions as needed
    dataset_path = args.dataset.replace("\\", "/")
    print("Dataset path: " + dataset_path)
    print("Output path: " + args.output)

    dataset_names = ['progan', 'biggan', 'cyclegan', 'eg3d', 'gaugan',  'stargan', 'stylegan', 'stylegan2', 'stylegan3', 
                 'dalle2', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'ldm_100', 'ldm_200', 'ldm_200_cfg',
                 'sd_512x512', 'sdxl', 'taming', 'deepfake', 'firefly', 'midjourney_v5', 'dalle3', 'faceswap']

    if args.variant == 'linearProb':
        eval_linear_prob(args, dataset_path, dataset_names, image_extensions, device)
    else:
        print('More methods coming soon')

    print('Evaluation completed!!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="linearProb", help="name of the adaptation method")
    parser.add_argument("--model", type=str, default="100k", help="name of linear probing model to evaluate")
    parser.add_argument("--dataset", type=str, default="", help="path to dataset")
    parser.add_argument("--output", type=str, default="", help="output directory to write results")
    
    args = parser.parse_args()
    main(args)