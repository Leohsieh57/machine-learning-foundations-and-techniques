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
           
        for depth in range(self.depth):
            NewKeyDict = {}
            for key in self.BinaryDict:
                if len(key) != self.depth - depth:
                    continue

                NewKey = key[:-1]
                if not NewKeyDict.get(NewKey):
                    NewKeyDict[NewKey] = []

                NewKeyDict[NewKey] += self.BinaryDict[key]

            for NewKey in NewKeyDict:
                self.BinaryDict[NewKey] = NewKeyDict[NewKey]

        for key in self.BinaryDict:
            self.BinaryDict[key] = sign(sum(self.BinaryDict[key]))

    def Predict(self, data):
        PredictionList = []
        for feat, _ in data.GetItemFull():
            OutcomeList = []
            for idx in self.PriorityList:
                model = self.ModelList[idx]
                x = feat[idx]
                OutcomeList.append(model.Predict(x))

            OutcomeString = OutcomeListToLabel(OutcomeList)+'_'
            for depth in range(self.depth):
                key = OutcomeString[:-(depth+1)]
                Prediction = self.BinaryDict.get(key)
                if Prediction != None:
                    PredictionList.append(Prediction)
                    break
            
        return PredictionList
    
def OutcomeListToLabel(OutcomeList):
    OutputString = ''
    for outcome in OutcomeList:
        OutputString += '1' if outcome == 1 else '0'

    return OutputString