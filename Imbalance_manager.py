import os
import pdb
import numpy as np
import scipy
from sklearn import svm
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


#class to apply a thecnique to manage the class imbalance
class Imbalance_classes:
    def __init__(self, xtrain,ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    def apply(self,string):
        #Description: the method resample the dataset with the selected method, giving as output the resampled dataset.
        #INPUT:  - string: used to choose the selected thecnique
        #OUTPUT: - X_train: new dataset resampled with the choosen algorithm.
        #        - y_train: labels corresponding to the new dataset.
        print 'Dimensional Reduction'
        print '---------------------'
        if string == 'SMOTE':
            tl = SMOTE(random_state=42)
        elif string == 'SMOTEENN':
            #3 neighbors for Edited nearest neighbors: Normal SMOTE: distance of 2 points of the class and new point in between
            tl = SMOTEENN(random_state=42)
        elif string == 'RandomOverSampling':
            tl = RandomOverSampler(random_state=42)
        elif string == 'None':
            return self.xtrain, self.ytrain
        X_train, y_train = tl.fit_sample(self.xtrain, self.ytrain)
        return X_train, y_train
