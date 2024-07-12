from __future__ import print_function
import csv
import glob
import os
import cv2
import torch
import random
import numpy as np
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import os.path as osp
from itertools import chain
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm.notebook import tqdm
from imgaug import augmenters as iaa
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class_labels = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor']

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
        iaa.Resize((224, 224)),
#         iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5), # horizontally flip
        iaa.OneOf([
            iaa.Affine(scale=1.5),
            iaa.Affine(rotate=20),
            iaa.Affine(translate_px=(-20, 20)),
            iaa.Cutout(fill_mode="constant", cval=0, nb_iterations=2, size=0.4)
        ]),
        ], random_order=True)
      
    def __call__(self, img):
        img = np.array(img).astype(np.uint8)
        img = self.aug.augment_image(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        return img

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def shuffle_list(inp_list):
    np.random.shuffle(inp_list)                
    np.random.shuffle(inp_list)          
    np.random.shuffle(inp_list)
    return inp_list

# Logic to get exactly the same amount of images from each category of Progan dataset e.g., 10000 (real + fake) images of airplane, 10000 images of car etc
def get_balanced_list_of_class_images(list_of_images, num_images_from_each_class, class_labels):
    subset_images = []
    class_count = {class_label: {'real': 0, 'fake': 0} for class_label in class_labels}
    image_seen = set()
    for image_path in list_of_images:
        for class_label in class_labels:
            if class_label in image_path and image_path not in image_seen:
                if 'n01443537' in image_path and class_count[class_label]['fake'] < num_images_from_each_class / 2:
                    subset_images.append(image_path)
                    class_count[class_label]['fake'] += 1
                    image_seen.add(image_path)
                elif 'n01440764' in image_path and class_count[class_label]['real'] < num_images_from_each_class / 2:
                    subset_images.append(image_path)
                    class_count[class_label]['real'] += 1
                    image_seen.add(image_path)
        if all(count['real'] >= num_images_from_each_class / 2 and count['fake'] >= num_images_from_each_class / 2 for count in class_count.values()):
            break
    return subset_images

def get_dataset(dataset_path, num_images_from_each_class, balance_classes):
    seed_everything(17)

    
    
    ## TO BE REMOVED
    # if dataset_name == 'progan':
    #     train_dir_real = '../CoOp/data/ImageNet_100k/images/train/n01440764/'
    #     train_dir_fake = '../CoOp/data/ImageNet_100k/images/train/n01443537/'
    #     validation_dir_real = '../CoOp/data/ImageNet_100k/images/val/n01440764/'
    #     validation_dir_fake = '../CoOp/data/ImageNet_100k/images/val/n01443537/'
    # elif dataset_name == 'sd':
    #     train_dir_real = '../CoOp/data/ImageNet/images/train/n01440764/'
    #     train_dir_fake = '../Datasets/ICMRDataset/train/stablediffusion/1_fake/'
    #     validation_dir_real = '../CoOp/data/ImageNet/images/val/n01440764/'
    #     validation_dir_fake = '../Datasets/ICMRDataset/validation/stablediffusion/1_fake/'
    # elif dataset_name == 'fewshot':
    #     train_dir_real = '../CoOp/data/ImageNet/images/train/n01440764/'
    #     train_dir_fake = '../CoOp/data/ImageNet/images/train/n01443537/'
    #     validation_dir_real = '../CoOp/data/ImageNet/images/val/n01440764/'
    #     validation_dir_fake = '../CoOp/data/ImageNet/images/val/n01443537/'
    ## TO BE REMOVED



    train_dir_real = dataset_path + '/images/train/n01440764/'
    train_dir_fake = dataset_path + '/images/train/n01443537/'
    validation_dir_real = dataset_path + '/images/val/n01440764/'
    validation_dir_fake = dataset_path + '/images/val/n01443537/'

    train_list_real = glob.glob(os.path.join(train_dir_real,'*'))
    train_list_fake = glob.glob(os.path.join(train_dir_fake,'*'))
    validation_list_real = glob.glob(os.path.join(validation_dir_real,'*'))
    validation_list_fake = glob.glob(os.path.join(validation_dir_fake,'*'))

    train_list_real = [path.replace('\\','/') for path in train_list_real]
    train_list_fake = [path.replace('\\','/') for path in train_list_fake]
    validation_list_real = [path.replace('\\','/') for path in validation_list_real]
    validation_list_fake = [path.replace('\\','/') for path in validation_list_fake]

    train_list_real = shuffle_list(train_list_real)
    train_list_fake = shuffle_list(train_list_fake)
    validation_list_real = shuffle_list(validation_list_real)
    validation_list_fake = shuffle_list(validation_list_fake)
    
    all_training_images = []
    all_training_images = train_list_real
    all_training_images.extend(train_list_fake)
    
    all_validation_images = []
    all_validation_images = validation_list_real
    all_validation_images.extend(validation_list_fake)

    all_training_images = shuffle_list(all_training_images)
    all_validation_images = shuffle_list(all_validation_images)

    # Logic to keep exactly same amount of images from each class
    if balance_classes:
        print('in balance')
        subset_training_images = []
        subset_validation_images = []
        subset_training_images = get_balanced_list_of_class_images(all_training_images, num_images_from_each_class, class_labels)
        subset_validation_images = get_balanced_list_of_class_images(all_validation_images, num_images_from_each_class, class_labels)
        all_training_images = shuffle_list(subset_training_images)
        # all_validation_images = shuffle_list(subset_validation_images)
    return all_training_images, all_validation_images

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
        iaa.Resize((224, 224)),
#         iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5), # horizontally flip
        iaa.OneOf([
            iaa.Affine(scale=1.5),
            iaa.Affine(rotate=20),
            iaa.Affine(translate_px=(-20, 20)),
            iaa.Cutout(fill_mode="constant", cval=0, nb_iterations=2, size=0.4)
        ]),
#         iaa.OneOf([
#             iaa.JpegCompression(compression=(60, 70)),
#             iaa.GaussianBlur((0, 1.0)),
#             iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.3),
#             iaa.Multiply((0.5, 1.0), per_channel=0.2),
#             iaa.Cutout(fill_mode="constant", cval=0, nb_iterations=1, size=0.5)
#                 ])
        ], random_order=True)
      
    def __call__(self, img):
        img = img.convert("RGB")
        img = np.array(img).astype(np.uint8)
        img = self.aug.augment_image(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img)
        return img


def get_transforms():
    BICUBIC = InterpolationMode.BICUBIC
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    clip_train_transforms = transforms.Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    clip_val_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    test_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=BICUBIC),
            transforms.ToTensor(),
        ])
    transforms_imgaug = ImgAugTransform()
    return train_transforms, val_transforms, clip_train_transforms, clip_val_transforms, test_transforms, transforms_imgaug

class DeepFakeSet(Dataset):
    def __init__(self, file_list, transform=None):

        self.file_list = file_list
        self.transform = transform
        self.to_img = ToPILImage()

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split('/')[-2]
        label = 1 if label == "n01443537" else 0
        return img_transformed, label
    
def get_class_based_training_data(class_name_list, all_training_images, all_validation_images):
    training_selected_images_list = []
    validation_selected_images_list = []
    for image_path in all_training_images:
        if any(class_name in image_path.split('/')[-1] for class_name in class_name_list):
            training_selected_images_list.append(image_path)
    for image_path in all_validation_images:
        if any(class_name in image_path.split('/')[-1] for class_name in class_name_list):
            validation_selected_images_list.append(image_path)
    return training_selected_images_list, validation_selected_images_list