import numpy as np

def Eout(PredictionList, LabelList):
    EoutList = [pred != label for pred, label in zip(PredictionList, LabelList)]
    return np.average(EoutList)