from __future__ import print_function
import clip
import timm
import torch
import random
import numpy as np
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import os.path as osp
from itertools import chain
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

from os import path
import sys

# sys.path.append(path.abspath('./beit/'))
# from beit.getBEIT import createBEIT, load_model
# device = 'cuda'

class LinearClassifier(torch.nn.Module):
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        torch.set_default_dtype(torch.float16)
        self.num_labels = num_labels
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)
        
class clipmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor, self.preprocess = clip.load("ViT-L/14", device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        # self.fc = nn.Linear(768, 2)
        self.fc = LinearClassifier(768, 2)

    def forward(self, x):
        # with torch.no_grad():
        intermediate_output = self.feature_extractor.encode_image(x)
        output = self.fc(intermediate_output)
        return output
    
# class beitmodel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.feature_extractor = createBEIT()
#         self.checkpoint_file='./beit/pretrained_weights/beit_large_patch16_224_pt22k_ft22k.pth'
#         load_model(model=self.feature_extractor, checkpoint_file=self.checkpoint_file, model_key='model|module|', model_prefix="")
#         self.classifier = LinearClassifier(1024, 2)
        
#     def forward(self, x):
#         with torch.no_grad():
#             intermediate_output = self.feature_extractor(x)
#         output = self.classifier(intermediate_output)
#         return output
    
class dinov2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.classifier = LinearClassifier(1024, 2)
        
    def forward(self, x):
        with torch.no_grad():
            intermediate_output = self.feature_extractor(x)
        output = self.classifier(intermediate_output)
        return output

    
class CLIPModelOhja(nn.Module):
    def __init__(self):
        super(CLIPModelOhja, self).__init__()
        self.model, self.preprocess = clip.load("ViT-L/14", device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        features = self.model.encode_image(x) 
        return self.fc(features)