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
from lib import preprocessor as prep
import csv
import multiprocessing as mp
import copy
import threading

final_dict, train_dict, test_dict = None, None, None

def read_file(fn):
    with open(fn, mode='r') as test:
        reader = csv.reader(test)
        rowList = [row[1:] for row in reader]
        header = rowList.pop(0)
        return header
        
def get_type_list(ls):
    type_dict = {}
    for mem in ls:
        type_dict[mem] = 0
            
    for idx, key in enumerate(type_dict):
        type_dict[key] = idx
        

    return type_dict

def list_is_float(dt):
    for key in dt:
        if key != 'Nan':
            if not isfloat(key):
                return False
    return True
            
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def dict_of_type_to_one_hot(args):
    #ls_dt is a list of dicts indicates heads to encodings(int)
    dt, head = args
    type_to_one_hot = {}
    for key in dt:
        one_hot = np.zeros(len(dt))
        one_hot[dt[key]] = 1
        type_to_one_hot[key] = one_hot
        
    return type_to_one_hot, head
            
def GetDictOfDict(fnTrain = 'data/train.csv', fnTest = 'data/test.csv'):    
    train_head = read_file(fnTrain)
    test_head = read_file(fnTest)
    train_addition = [head for head in train_head if head not in test_head]
    #print(train_addition)
    train_dict = pd.read_csv(fnTrain).fillna('NaN')
    test_dict = pd.read_csv(fnTest).fillna('NaN')
    pool = mp.Pool(mp.cpu_count())
    mp_list = [list(train_dict[hd])+list(test_dict[hd]) for hd in test_head]
    list_of_bool_if_hd_is_float = pool.map(list_is_float, mp_list)
    assert(len(mp_list)==len(list_of_bool_if_hd_is_float))
    zip_list = zip(mp_list, list_of_bool_if_hd_is_float)
    mp_list = [mem for mem, log in zip_list if not log]
    list_of_dict_type_to_encoding = pool.map(get_type_list, mp_list)
    
    assert(len(test_head)==len(list_of_bool_if_hd_is_float))
    zip_list = zip(test_head, list_of_bool_if_hd_is_float)
    list_of_non_float_head = [mem for mem, log in zip_list if not log]
    #print(list_of_dict_type_to_encoding)
    #print(list_of_non_float_head)
    
    assert(len(list_of_dict_type_to_encoding)==len(list_of_non_float_head))
    zip_list = zip(list_of_dict_type_to_encoding, list_of_non_float_head)
    list_of_head_and_one_hot_dict = pool.map(dict_of_type_to_one_hot, zip_list)
    final_dict = {}
    for type_to_one_hot, head in list_of_head_and_one_hot_dict:
        final_dict[head] = type_to_one_hot
    #print(final_dict['hotel'])
    #print(final_dict['arrival_date_month'])
    return final_dict, train_dict, test_dict

def mpFuncTest(i):
    vector_list = []
    global test_dict
    global final_dict
    for key in test_dict:
        mem = test_dict[key][i]
        dict_en = final_dict.get(key)
        if dict_en:
            vector_list += dict_en[mem].tolist()
        else:
            a = -1 if mem =='NaN' else mem
            vector_list += [float(a)]
    return vector_list

def mpFuncTrain(i):
    vector_list = []
    global train_dict
    global final_dict
    for key in test_dict:
        mem = train_dict[key][i]
        dict_en = final_dict.get(key)
        if dict_en:
            vector_list += dict_en[mem].tolist()
        else:
            a = -1 if mem =='NaN' else mem
            vector_list += [float(a)]
    return vector_list

def GetFeatureVectorsAndLabels(fnTrain = 'data/train.csv', fnTest = 'data/test.csv'):
    global final_dict
    global train_dict
    global test_dict
    final_dict, train_dict, test_dict = prep.GetDictOfDict(fnTrain, fnTest)
    pool = mp.Pool(mp.cpu_count())
    data_size = 0            
    for idx, key in enumerate(train_dict):
        if idx:
            assert(len(train_dict[key] == data_size))
        else:
            data_size = len(train_dict[key])
    
    train_data_tensor = pool.map(mpFuncTrain, list(range(data_size)))
                                 
    for idx, key in enumerate(test_dict):
        if idx:
            assert(len(test_dict[key] == data_size))
        else:
            data_size = len(test_dict[key])
                                 
    test_data_tensor = pool.map(mpFuncTest, list(range(data_size)))
    label_adr = [[adr] for adr in train_dict['adr']]
    label_can = [adr for adr in train_dict['is_canceled']]
    rev_dict = {}
    for rev in train_dict['reservation_status']:
        if not rev_dict.get(rev):
            rev_dict[rev] = 0
    for idx, key in enumerate(rev_dict):
        rev_dict[key] = idx
    
    dict_final_rev_label = {}
    for key1 in rev_dict:
        for key2 in label_can:
            dict_final_rev_label[key1+', '+str(key2)] = 0
            
    for idx, key in enumerate(dict_final_rev_label):
        dict_final_rev_label[key] = idx
    print('Encoding completed\n')
    print('Using the Following Notation: \n' ,dict_final_rev_label)
    zipped = zip(train_dict['reservation_status'], train_dict['is_canceled'])
    label_rev = [[dict_final_rev_label[key1+', '+str(key2)]] for key1, key2 in zipped]
    #label_res = [final_dict['reservation_status'][adr] for adr in train_dict['reservation_status']]
    label_adr = torch.tensor(label_adr)
    label_rev = torch.tensor(label_rev)
    #label_res = torch.tensor(label_res)
    return torch.tensor(train_data_tensor), torch.tensor(test_data_tensor), label_adr, label_rev, #label_res