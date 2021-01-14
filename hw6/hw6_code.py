import numpy as np
import multiprocessing as mp
from DataSet import DataSet 
from DecisionStumpModel import DecisionStumpModel
from DecisionTree import DecisionTree

def mpEin(model):
    Ein = model.GetOptParams()
    return Ein, model

if __name__ == '__main__':
    train, test = DataSet('test.dat'), DataSet('train.dat')

    # question14
    print('question 14')
    ModelList = [DecisionStumpModel(feat, label) for feat, label in train.GetFeatFull()]
    pool = mp.Pool(mp.cpu_count())
    EinModelList = pool.map(mpEin, ModelList)
    EinList = [Ein for Ein, model in EinModelList]
    ModelList = [model for Ein, model in EinModelList]
    for model in ModelList:
        print(model.s, model.theta)
    PriorityList = [(Ein, idx) for idx, Ein in enumerate(EinList)]
    PriorityList = [idx for Ein, idx in sorted(PriorityList)]
    Tree = DecisionTree(train, ModelList)
    Tree.SetPriority(PriorityList)
    Tree.ConstructTree()
    print('a')