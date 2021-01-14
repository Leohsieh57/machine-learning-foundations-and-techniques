import numpy as np

class DataSet:
    def __init__(self, fn):
        file = open(fn)
        data = [[float(mem) for mem in line[:-2].split(' ')] for line in file]
        self.data = np.array(data)
        self.DataCnt = self.data.shape[0]
        self.FeatCnt = self.data.shape[1]-1
        
    def GetItem(self, idx):
        assert idx < self.DataCnt
        label = self.data[idx][-1]
        feat = self.data[idx][:-1]
        return feat, label

    def GetItemFull(self):
        return [self.GetItem(idx) for idx in range(self.DataCnt)]
    
    def GetFeat(self, idx): #ith feat
        assert idx < self.FeatCnt 
        FeatList = self.data[:,idx]
        LabelList = self.data[:,-1]
        return FeatList, LabelList
    
    def GetFeatFull(self):
        return [self.GetFeat(idx) for idx in range(self.FeatCnt)]