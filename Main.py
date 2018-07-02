import numpy as np
import pdb
import scipy
from sklearn.model_selection import KFold
import copy
import scipy.io
#myclasses
from DR import DimReduction
from Classification import Classification_methods
from Opening_design import GUI_design
from Imbalance_manager import Imbalance_classes

#Nfeatures is the number of final features after the reduction,
#both for future selection methods and feature reduction
Nfeatures = 600
#number of folds for the K-fold cross validation
NFOLDS = 10
#display methods: the algorithm either takes arguments as parameters in the launching command,
#or directly from input/output in the first part of the program.
displayer = GUI_design()
Arguments = displayer.Pipeline_construction()
Feature_reduction = Arguments[0]
Imbalance_model = Arguments[1]
Classification_choice = Arguments[2]
np.random.seed(0)

#loading of the dataset
dataset = scipy.io.loadmat('dataset.mat')
X = dataset['Transcriptome'][1:,:]
Y2 = dataset['Y'][1:]
feat_name = dataset['Transcriptome_labels']
del dataset
#we eclude all the Healty patients or patient with Not present labels
Y = np.zeros([len(Y2)])
for i in range(len(Y)):
    if Y2[i] == 'Basal-like   ': Y[i] = 1
    if Y2[i] == 'Normal-like  ': Y[i] = 2
    if Y2[i] == 'Luminal A    ': Y[i] = 3
    if Y2[i] == 'Luminal B    ': Y[i] = 4
    if Y2[i] == 'HER2-enriched': Y[i] = 5
    if Y2[i] == 'Healty       ': Y[i] = 6
    if Y2[i] == 'NA           ': Y[i] = 6
    if Y2[i] == 'Not present  ': Y[i] = 6

X = X[Y<6, :]
Y = Y[Y<6]

#remove features with all 0
X = X[:, sum(X,0)!=0]
#Unsupervised pipeline: we force reduction to PCA and not class imbalance to apply the clustering:
#In this case we don't perform Kfold, because we don't have the labels and it is useless.
# We use the labels at the end to decide which cluster assign to each label and calculate the accuracy and the other scores.
if Classification_choice == 'kMeans' or Classification_choice == 'HierarchicalClustering':
    FeatR = DimReduction(mdl_type = Feature_reduction, dim_out = Nfeatures)
    X_train = FeatR.Dataset_reduction(X, supervised = False)
    classifier = Classification_methods(X,ytest=Y)
    Accuracy, F1_score,BER = classifier.Classification_start(Classification_choice)
else:
    #k-fold cross validation
    kf = KFold(n_splits=NFOLDS,shuffle=True, random_state=2)
    Accuracy = 0
    F1_score = 0
    BER = 0
    i=1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        ##Feature reduction
        FeatR = DimReduction(mdl_type = Feature_reduction, dim_out = Nfeatures)
        X_train, X_test = FeatR.Dataset_reduction(X_train = X_train,supervised = True,X_test= X_test, labels =y_train)
        ##Class Imbalance
        model0 = Imbalance_classes(X_train, y_train)
        X_train, y_train = model0.apply(Imbalance_model)
        np.random.seed(0)
        #Learning Step
        classifier = Classification_methods(X_train, y_train,X_test,y_test)
        Acc, F1,BER_int = classifier.Classification_start(Classification_choice)
        print 'Summary of fold {:.0f}'.format(i)
        print '-----------------'
        print 'Accuracy = {:.2f} \nF1 score = {:.2f} \nBER = {:.2f}'\
        .format(Acc,F1,BER_int)
        i = i + 1
        Accuracy = Accuracy + Acc
        F1_score = F1_score + F1
        BER = BER + BER_int
    #metrics to evaluate the algorithm. See the corresponding class.
    Accuracy = Accuracy/float(NFOLDS)
    F1_score = F1_score/float(NFOLDS)
    BER = BER/float(NFOLDS)
print 'Final Summary'
print '-------------'
print 'Accuracy = {:.2f} \nF1 score = {:.2f} \nBER = {:.2f}'\
.format(Accuracy,F1_score,BER)
