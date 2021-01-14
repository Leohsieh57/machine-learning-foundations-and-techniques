import numpy as np
import multiprocessing as mp
from DataSet import DataSet 
from DecisionStumpModel import DecisionStumpModel

def mpEin(model):
    return model.GetOptParams()

if __name__ == '__main__':
    train, test = DataSet('test.dat'), DataSet('train.dat')

    # question14
    ModelList = [DecisionStumpModel(feat, label) for feat, label in train.GetFeatFull()]
    pool = mp.Pool(mp.cpu_count())
    print(pool.map(mpEin, ModelList))