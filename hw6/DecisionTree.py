import numpy as np
from DataSet import DataSet 

class DecisionTree:
    def __init__(self, data):
        self.data = data
        self.DataCnt = self.data.shape[0]
        self.FeatCnt = self.data.shape[1]-1
        
    def GetItem(self, idx):
        assert(idx < self.DataCnt)
        label = self.data[idx][-1]
        feat = self.data[idx][:-1]
        return feat, label
    
    def GetFeat(self, idx): #ith feat
        assert(idx < self.FeatCnt)
        FeatList = self.data[:,idx]
        LabelList = self.data[:,-1]
        return FeatList, LabelList
    
    def GetFeatFull(self):
        return [self.GetFeat(idx) for idx in range(self.FeatCnt)]

class BinaryTree:
    def __init__(self, parent):
        self.data = data