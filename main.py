
from cifar import loadcifar, cifar
from reuter import loadreuter, Reuter
import numpy as np
import pandas as pd
#from utils.classifier import *
import argparse



def main():

    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument('-DataName', action='store', dest='DataName', default='cifar')
    parser.add_argument('-FromLanguage', action='store', dest='FromLanguage', default='EN')
    parser.add_argument('-ToLanguage', action='store', dest='ToLanguage', default='FR')
    args = parser.parse_args()
    learner = OLDS(args)
    learner.train()

class OLDS:
    def __init__(self, args):
        '''
            Data is stored as list of dictionaries.
            Label is stored as list of scalars.
        '''
        self.datasetname = args.DataName
        self.FromLan = args.FromLanguage
        self.ToLan = args.ToLanguage

    def train(self):
        if self.datasetname == 'cifar':
            print('trainning starts')
            x_S1, x_S2, y_S1, y_S2 = loadcifar()
            train = cifar(x_S1, y_S1, x_S2, y_S2, 50000, 5000)
            train.T_2()
        else:
            if self.FromLan =='EN':
                self.samplesize = 18758
                self.overlap = 2758
                self.dimension1 = 2000
                if self.ToLan == 'FR':
                    self.dimension2 = 2500
                elif self.ToLan == 'IT':
                    self.dimension2 = 1500
                else:
                    self.dimension2 = 1000
            else:
                self.samplesize = 26648
                self.overlap = 3648
                self.dimension1 = 2000
                if self.ToLan == 'IT':
                    self.dimension2 = 1500
                else:
                    self.dimension2 = 1000

            x_S1, x_S2, y_S1, y_S2 = loadreuter(self.FromLan,self.ToLan,
                                                self.samplesize, self.dimension, self.dimension2)
            train = Reuter(x_S1, y_S1, x_S2, y_S2, self.samplesize, self.samplesize,
                           self.dimension1,self.dimension2,self.FromLan,self.ToLan)
            train.T_2()



if __name__ == '__main__':
   main()


