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

device = 'cuda'
def train_model(model, model_name, epochs, learning_rate, train_loader, valid_loader):
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
    # scheduler
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for data, label in (train_loader):
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

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in (valid_loader):
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
        torch.cuda.empty_cache()
    return model