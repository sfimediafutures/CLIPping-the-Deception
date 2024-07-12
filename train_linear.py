from __future__ import print_function
import glob
import clip
from itertools import chain
import os
import cv2
import random
# import zipfile
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# from functools import reduce
import torch.nn as nn
# from einops import rearrange, repeat
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from imgaug import augmenters as iaa
# from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
# from skimage import io, img_as_float
import timm
from torchvision.transforms import ToPILImage
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import csv
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score, precision_recall_curve
import argparse

# Local imports
from utils import class_labels, get_dataset, get_transforms, DeepFakeSet, get_class_based_training_data
from models import clipmodel, dinov2, CLIPModelOhja
from trainer import train_model

device = 'cuda'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(args):

    print("Starting Training!")
    seed_everything(seed=17)

    # Logic to get exactly the same amount of images from each category of Progan dataset e.g., 10000 (real + fake) images of airplane, 10000 images of car etc
    if args.train_strategy == 'fewshot':
        num_images_from_each_class = 32
    elif args.train_strategy == '100k':
        num_images_from_each_class = 10000
    elif args.train_strategy == '80k':
        num_images_from_each_class = 4000
    elif args.train_strategy == '60k':
        num_images_from_each_class = 3000
    elif args.train_strategy == '40k':
        num_images_from_each_class = 2000
    elif args.train_strategy == '20k':
        num_images_from_each_class = 1000

    # balance_classes will be set to True if we need subset of images, lets say if we need 250 image from each class for shorter training
    all_training_images, all_validation_images = get_dataset(args.dataset_path, num_images_from_each_class, balance_classes=True)

    print(f"Training Data: {len(all_training_images)}")
    print('***********************************')
    print('***********************************')
    print(f"Validation Data: {len(all_validation_images)}")

    train_transforms, val_transforms, clip_train_transforms, clip_val_transforms, test_transforms, transforms_imgaug = get_transforms()

    train_data = DeepFakeSet(all_training_images, transform=transforms_imgaug)
    valid_data = DeepFakeSet(all_validation_images, transform=clip_val_transforms)

    batch_size = 16

    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)

    print(len(train_data), len(train_loader))
    print(len(valid_data), len(valid_loader))

    # model = CLIPModelOhja()
    model = clipmodel()
    model.to(device)

    print('Turning off gradients in both the image and the text encoder')
    for name, param in model.named_parameters():
        if 'fc.linear.weight' not in name and 'fc.linear.bias' not in name:
            param.requires_grad_(False)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable Parameters: ', str(params))
    
    epochs = 2
    lr = 3e-3
    # gamma = 0.7
    warmup_epochs = 0

    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer = optim.SGD(model.fc.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    # Warm-up epoch with a different learning rate
    warmup_optimizer = optim.SGD(model.fc.parameters(), lr=0.0001)

    
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            optimizer = warmup_optimizer
        else:
            optimizer = optimizer
        epoch_loss = 0
        epoch_accuracy = 0

        # Keep track of intermediate statistics for every 10 batches
        running_loss = 0.0
        running_accuracy = 0.0
        print_every = 10
        
        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

            running_accuracy += acc
            running_loss += loss.item()

            # Print accuracy and loss after every 5 processed batches
            if (batch_idx + 1) % print_every == 0:
                avg_running_accuracy = running_accuracy / print_every
                avg_running_loss = running_loss / print_every
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_running_loss:.4f}, Acc: {avg_running_accuracy:.4f}')
                running_accuracy = 0.0
                running_loss = 0.0

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)
    #             print(data.shape)aa
                with torch.cuda.amp.autocast():
                    val_output = model(data)
                val_loss = criterion(val_output, label)
                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )

    torch.save(model.state_dict(), str(args.output_path) + 'CLIP_linear_prob_' + str(epoch+1) + '.pth')
    torch.cuda.empty_cache()
    print("Training Finsihed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 100k real and 100k fake images, or 80k real and 80k fake images and so on
    parser.add_argument("--train_strategy", type=str, choices=["fewshot", "100k", "80k", "60k", "40k", "20k"], default="100k", help="name of training strategy e.g., fewshot, 100k, 80k, 60k, 40k, 20k")
    parser.add_argument("--dataset_path", type=str, default="", help="path to dataset")
    parser.add_argument("--output_path", type=str, default="./linearProbingOutputs/", help="output directory to write results")
    
    args = parser.parse_args()
    main(args)