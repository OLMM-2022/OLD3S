import math
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.nn.functional import normalize
from torchvision.transforms.functional import scale





class LoadReuter():
    '''    Instances X Features
        EN_EN: 18758 X 21531 (origin)
        EN_FR: 18758 X 24893
        EN_GR: 18758 X 34215
        EN_IT: 18758 X 15506
        EN_SP: 18758 X 11547
    '''

    def __init__(self, fromLang, dataset_path, seed=42):
        print('LOAD Reuter from: ', str(fromLang))
        self.fromLang = fromLang
        self.dataset_path = dataset_path
        dataOrigin = pd.read_csv(
            self.dataset_path + 'Reuter/' + str(fromLang) + '/Index_' + str(fromLang) + '-' + str(fromLang),
            header=None).values
        self.data, self.label = ReuterPreprocess(dataOrigin, seed=seed)

    def normlize(self, norm='l1'):
        if norm == 'l2':
            self.data = normalize(self.data, norm='l2', axis=0)
        elif norm == 'L1':
            self.data = normalize(self.data, norm='l1', axis=0)
        elif norm == 'scale':
            self.data = scale(self.data)

    def translate(self, toLang, seed=42):
        fromLang = self.fromLang
        print('LOAD Reuter from: ', str(fromLang), "to", str(toLang))

        dataTrans = pd.read_csv(
            self.dataset_path + 'Reuter/' + str(toLang) + '/Index_' + str(self.fromLang) + '-' + str(toLang),
            header=None).values
        data, label = ReuterPreprocess(dataTrans, seed=seed)
        return data, label

    def enlargerEvo(self, toLang=None, overlapRate=0.1, seed=42):
        print('to: ', str(toLang))
        dataTrans = pd.read_csv(
            self.dataset_path + 'Reuter/' + str(toLang) + '/Index_' + str(self.fromLang) + '-' + str(toLang),
            header=None).values
        dataT, labelT = ReuterPreprocess(dataTrans, seed=seed, startkey=len(self.data))
        overlap_start = math.floor(len(self.data) * (1 - overlapRate))
        for i in range(overlap_start, len(self.data)):
            self.data[i].update(dataT.pop(overlap_start))
            labelT = np.delete(labelT, overlap_start, 0)
        self.data.extend(dataT)
        self.label = np.append(self.label, labelT, axis=0)


def ReuterPreprocess(data, seed=42, startkey=0):
    dataNew = []
    label = []
    for i, row in enumerate(data):
        # print(row)
        string = row[0]
        string = string.split(' ')
        label.append(string.pop(0))
        tempDict = {}
        for _, dict in enumerate(string):
            # print(dict)
            k, p, v = dict.partition(':')
            if p != ':':
                continue
            stringDict = {startkey + int(k): float(v)}

            tempDict.update(stringDict)

        dataNew.append(tempDict)
    y = pd.DataFrame(label)
    y = y.replace('C15', 0)
    y = y.replace('CCAT', 1)
    y = y.replace('E21', 2)
    y = y.replace('ECAT', 3)
    y = y.replace('GCAT', 4)
    y = y.replace('M11', 5)
    y = y.values.reshape(-1, 1)
    data, label = shuffle(dataNew, y, random_state=seed)

    return data, label
    # return dataNew, y


def addZERO(dataset, size):
    data = torch.zeros(size)
    i = 0
    for dict in dataset:
        for j in dict.keys():
            
            data[i][j - 1] = dict[j]
        i += 1
    return data


