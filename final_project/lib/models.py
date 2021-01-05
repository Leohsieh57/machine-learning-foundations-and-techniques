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
import pandas as pd
import matplotlib.pyplot as plt
from lib import preprocessor as prep
import csv
import multiprocessing as mp
import copy
import threading
import random

class Classifier(nn.Module):
    def __init__(self, input_size = 258):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 180),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(180, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 6)
        )

    def forward(self, x):
        return self.fc(x)
    
class ClassifierLinear(nn.Module):
    def __init__(self, input_size = 258):
        super(ClassifierLinear, self).__init__()
        self.fc = nn.Linear(input_size, 6)

    def forward(self, x):
        return self.fc(x)

