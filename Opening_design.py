import os
import pandas as pd
import pdb
import numpy as np
import math
import scipy
import argparse

class GUI_design:
    def Initial_Display(self):
        print '\nWelcome to the GABAB tool for breast cancer recognition'
        print '-------------------------------------------------------\n'
    def Pipeline_definition(self, string):
        #Description: This function takes as input a string to define a part of the pipeline taken from the corresponding List
        #             The functions is choosen using the keyboard..
        #INPUT:  - string:define which part of the pipeline you are setting
        #OUTPUT: - List_used[i-1]: returns the specific thecnique choosen for the target part of the pipeline.
        String2Part = {'features': 'The algorithms for Features reduction are the following:\n', \
                       'classes': 'The algorithms to manage the class imbalance are the following:\n', \
                       'learning': 'Choose between supervised and unsupervised analysis:\n',\
                        'supervised': 'The supervised methods implemented are the following ones:\n',\
                        'unsupervised': 'The unsupervised methods implemented are the following ones:\n'}
        Feat = ['PCA',
                'LDA',
                'GA']
        Classes = ['SMOTE',
                'SMOTEEN',
                'RandomOverSampling',
                'None']
        Learnings = ['Supervised',
                'Unsupervised']
        Supervised = ['SVC',
                'kNN',
                'Random_Forest']
        Unsupervised = ['kMeans',
                'HierarchicalClustering']
        String2list = {'features': Feat, \
                       'classes': Classes, \
                       'learning': Learnings,\
                        'supervised': Supervised,\
                        'unsupervised': Unsupervised}
        String_to_print = String2Part[string]
        List_used = String2list[string]
        for i in range(len(List_used)):
            String_to_print = String_to_print +str(i+1)+')'+List_used[i]+'\n'
        i = input(String_to_print)
        return List_used[int(i-1)]

    def Pipeline_construction(self):
        #Description: This function constructs the target pipeline. It can be either construct using the build command with the
        #             parameters or the user interface, simply following the commands shown on the terminal. If you insert
        #             wrong parameters in the build command, the terminal will show an error.
        #INPUT:
        #OUTPUT: - Arguments: composed by the 3 elements of the pipeline, adressed by their name
        self.Initial_Display()
        parser = argparse.ArgumentParser()
        parser.add_argument("--fast", default = False, type = bool, dest = 'fast', help="Set to true if you want to avoid the decision of all the steps by hand")
        parser.add_argument("--reduction", default = 'PCA', dest = 'reduction', help="Feature_reduction or selection algorithm.  Possible methods: PCA,LDA,GA(Genetic Algorithm)")
        parser.add_argument("--imbalance", default = 'None', dest = 'imbalance', help="Class imbalance management thecniques. Possible methods: SMOTE,SMOTEENN,RandomOverSampling,None")
        parser.add_argument("--supervised", default = 'SVC', dest = 'supervised', help="Supervised methods to set.  Possible methods: SVC,kNN,Random_Forest")
        parser.add_argument("--unsupervised", default = None, dest = 'unsupervised', help="Unsupervised method. If set the algorithm will use the unsupervised pipeline. Possible methods: kMeans,HierarchicalClustering")
        args = parser.parse_args()
        assert args.fast in [False, True]
        assert args.reduction in ['PCA','LDA', 'GA']
        assert args.imbalance in ['RandomOverSampling', 'SMOTE','SMOTEEN','None']
        assert args.supervised in ['SVC', 'kNN', 'Random_Forest']
        assert args.unsupervised in [None,'kMeans','HierarchicalClustering']
        if args.fast == True:
            if args.unsupervised == None:
                arguments = [args.reduction, args.imbalance, args.supervised]
            else:
                args.reduction = 'PCA'
                args.imbalance = 'None'
                arguments = [args.reduction, args.imbalance, args.unsupervised]
        else:
            type_learning = self.Pipeline_definition('learning')

            if type_learning == 'Supervised':
                args.reduction = self.Pipeline_definition('features')
                args.imbalance = self.Pipeline_definition('classes')
                args.supervised = self.Pipeline_definition('supervised')
                arguments = [args.reduction, args.imbalance, args.supervised]
            else:
                args.unsupervised = self.Pipeline_definition('unsupervised')
                args.reduction = 'PCA'
                args.imbalance = 'None'
                arguments = [args.reduction, args.imbalance, args.unsupervised]
        print 'Pipeline running: \n'+'Feature reductor: \t\t' + arguments[0] +\
                '\nClass imbalance management: \t' + arguments[1] +\
                '\nAlgorithm choosen: \t\t' + arguments[2]
        print '\n'
        return arguments
