import numpy as np
import multiprocessing as mp
from DataSet import DataSet 
from DecisionStumpModel import DecisionStumpModel

def f(x):
    return x**2

if __name__ == '__main__':
    train, test = DataSet('test.dat'), DataSet('train.dat')
    pool = mp.Pool(5)
    rel  = pool.map(f,[1,2,3,4,5,6,7,8,9,10])

    print(rel)