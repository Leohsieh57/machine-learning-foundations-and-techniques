import numpy as np
from DataSet import DataSet 
from DecisionStumpModel import sign

class DecisionTree:
    def __init__(self, data, ModelList):
        self.data = data
        self.depth = len(ModelList)
        self.ModelList = ModelList
        self.BinaryDict = {}
        self.PriorityList = None

    def SetPriority(self, PriorityList):
        assert len(PriorityList) == self.depth
        self.PriorityList = PriorityList

    def ConstructTree(self):
        assert self.PriorityList
        for feat, label in self.data.GetItemFull():
            OutcomeList = []
            for idx in self.PriorityList:
                model = self.ModelList[idx]
                x = feat[idx]
                OutcomeList.append(model.Predict(x))

            OutcomeString = OutcomeListToLabel(OutcomeList)
            if not self.BinaryDict.get(OutcomeString):
                self.BinaryDict[OutcomeString] = []

            self.BinaryDict[OutcomeString].append(label)
           
        for key in self.BinaryDict:
            self.BinaryDict[key] = sign(sum(self.BinaryDict[key]))

        print(self.BinaryDict)
    

def OutcomeListToLabel(OutcomeList):
    OutputString = ''
    for outcome in OutcomeList:
        OutputString += '1' if outcome == 1 else '0'

    return OutputString