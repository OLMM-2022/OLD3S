
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
    parser.add_argument('-beta', action='store', dest='beta', default=0.9)
    parser.add_argument('-learningrate', action='store', dest='learningrate', default=0.01)
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
        self.beta = args.beta
        self.learningrate = args.learningrate
    def train(self):
        if self.datasetname == 'cifar':
            print('trainning starts')
            x_S1, x_S2, y_S1, y_S2 = loadcifar()
            train = cifar(x_S1, y_S1, x_S2, y_S2, 50000, 5000,self.beta,self.learningrate)
            train.T_2()
        else:
            if self.FromLan =='EN':
                self.samplesize = 18758
                self.overlap = 2758
                self.dimension1 = 21531
                self.dimension1_pca = 2000
                if self.ToLan == 'FR':
                    self.dimension2 = 24893
                    self.dimension2_pca = 2500
                elif self.ToLan == 'IT':
                    self.dimension2 = 15506
                    self.dimension2_pca = 1500
                else:
                    self.dimension2 = 11547
                    self.dimension2_pca = 1000
            else:
                self.samplesize = 26648
                self.overlap = 3648
                self.dimension1 = 24893
                self.dimension1_pca = 2500
                if self.ToLan == 'IT':
                    self.dimension2 = 15503
                    self.dimension2_pca = 1500
                else:
                    self.dimension2 = 11547
                    self.dimension2_pca = 1000

            x_S1, x_S2, y_S1, y_S2 = loadreuter(self.FromLan,self.ToLan,
                                                self.samplesize, self.dimension1, self.dimension2)
            train = Reuter(x_S1, y_S1, x_S2, y_S2, self.samplesize, self.samplesize,
                           self.dimension1_pca,self.dimension2_pca,self.FromLan,self.ToLan,self.beta,self.learningrate)
            train.T_2()



if __name__ == '__main__':
   main()


