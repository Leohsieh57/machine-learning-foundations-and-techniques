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
import csv

def LoadData(fnTrain, fnTest):    
    dictTrain = pd.read_csv(fnTrain)
    header = None
    with open(fnTest, mode='r') as test:
        reader = csv.reader(test)
        rowList = [row[1:] for row in reader]
        header = rowList.pop(0)

    featList = [dictTrain[fn] for fn in header]
    zippedList = zip(dictTrain['arrival_date_year'], dictTrain['arrival_date_month'], dictTrain['arrival_date_day_of_month'])
    dateStrList = [str(y)+str(m)+str(d) for y, m, d in zippedList]
    encodedFeatList = []
    for feat in featList:
        encodedFeat = []
        if not isfloat(feat[0]):
            memDict = {}
            for mem in feat:
                if memDict.get(mem) == None:
                    memDict[mem] = len(memDict)
                    
                encodedFeat.append(memDict[mem])
        else:
            encodedFeat = [float(mem) for mem in feat]
        
        encodedFeatList.append(encodedFeat)
        
    return np.array(encodedFeatList).T, dateStrList
    
    
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
def ClusterByDate(fnTrain, fnTest):
    encodedFeatList, dateStrList = LoadData(fnTrain, fnTest)
    assert(len(encodedFeatList) == len(dateStrList))
    dateStrDict = {}
    DataStrList = []
    for jumonVec, dateStr in zip(encodedFeatList, dateStrList):
        if dateStrDict.get(dateStr) == None:
            dateStrDict[dateStr] = []
            DataStrList.append(dateStr)
        dateStrDict[dateStr].append(jumonVec)
    
    return dateStrDict, DataStrList

def GetLargestJumonCnt(fnTrain, fnTest):
    dateStrDictTest, DataStrListTest = ClusterByDate('data/test.csv', 'data/test.csv')
    dateStrDictTrain, DataStrListTrain = ClusterByDate('data/train.csv', 'data/test.csv')
    maxSize = 0
    for DataStr in DataStrListTest:
        maxSize = max(maxSize, len(dateStrDictTest[DataStr]))
        
    for DataStr in DataStrListTrain:
        maxSize = max(maxSize, len(dateStrDictTrain[DataStr]))
    return maxSize

def GetFeatureMatrix(fnTrain, fnTest):
    a, b = ClusterByDate(fnTrain, fnTest)
    c, d = ClusterByDate(fnTest, fnTest)
    TrainFinalArray, TestFinalArray = [], []
    for date in b:
        vecList = a[date]
        while(len(vecList)<448):
            vecList.append(np.zeros(28))
        TrainFinalArray.append(vecList)
    for date in d:
        vecList = c[date]
        while(len(vecList)<448):
            vecList.append(np.zeros(28))
        TestFinalArray.append(vecList)
            
    return torch.tensor(TrainFinalArray), torch.tensor(TestFinalArray)

def LoadLabels(fn):
    dictLabel = pd.read_csv(fn)
    return torch.tensor([[label] for label in dictLabel['label']])
    