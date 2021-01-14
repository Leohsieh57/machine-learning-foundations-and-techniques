import numpy as np
import multiprocessing as mp
from DataSet import DataSet 
from DecisionStumpModel import DecisionStumpModel
from DecisionTree import DecisionTree
import Functions as func

def mpEinModel(model):
    Ein = model.GetOptParams()
    return Ein, model

if __name__ == '__main__':
    #set up
    pool = mp.Pool(mp.cpu_count())
    train, test = DataSet('test.dat'), DataSet('train.dat')

    # question14
    print('question 14')
    ModelList = [DecisionStumpModel(feat, label) for feat, label in train.GetFeatFull()]
    EinModelList = pool.map(mpEinModel, ModelList)
    # sort index with Ein
    EinList = [Ein for Ein, model in EinModelList]
    ModelList = [model for Ein, model in EinModelList]
    PriorityList = [(Ein, idx) for idx, Ein in enumerate(EinList)]
    PriorityList = [idx for Ein, idx in sorted(PriorityList)]
    # construct decision tree
    Tree = DecisionTree(train, ModelList)
    Tree.SetPriority(PriorityList)
    Tree.ConstructTree()
    # predict
    PredictList = Tree.Predict(test)
    LabelList = test.GetLabels()
    Eout = func.Eout(PredictList, LabelList)
    print('Ans: Eout =', Eout)