import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import torchvision.models as models
import copy
from torchvision.utils import save_image
import PIL
import skimage.io
import multiprocessing as mp
import time
import pandas as pd
import matplotlib.pyplot as plt
import random

def GetValidationVectors(root):
    val_x = torch.load(os.path.join(root, 'test_data_tensor.pt')).numpy()
    return val_x

def LoadDataRev(root):
    train_x = torch.load(os.path.join(root, 'train/data.pt')).numpy()
    train_y = torch.load(os.path.join(root, 'train/rev.pt')).numpy().reshape(-1,)
    test_x = torch.load(os.path.join(root, 'test/data.pt')).numpy()
    test_y = torch.load(os.path.join(root, 'test/rev.pt')).numpy().reshape(-1,)
    return train_x, train_y, test_x, test_y

def LoadDataAdr(root):
    train_x = torch.load(os.path.join(root, 'train/data.pt')).numpy()
    train_y = torch.load(os.path.join(root, 'train/adr.pt')).numpy().reshape(-1,)
    test_x = torch.load(os.path.join(root, 'test/data.pt')).numpy()
    test_y = torch.load(os.path.join(root, 'test/adr.pt')).numpy().reshape(-1,)
    return train_x, train_y, test_x, test_y
    
class HotelReservationData(Dataset):
    def __init__(self, root):
        self.root = root
        self.data = torch.load(os.path.join(self.root, 'data.pt'))
        self.adr = torch.load(os.path.join(self.root, 'adr.pt'))
        self.rev = torch.load(os.path.join(self.root, 'rev.pt'))
        assert(len(self.adr) == len(self.rev))
        assert(len(self.adr) == len(self.data))
        self.len = len(self.adr)
    
    def __getitem__(self, i):
        data = self.data[i]
        adr = float(self.adr[i])
        rev = float(self.rev[i] != 0)
        return data, adr, torch.tensor([rev])
    
    def __len__(self):
        return self.len