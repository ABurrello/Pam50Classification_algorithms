import os
import pdb
import numpy as np
import scipy
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class Classification_methods:
    def __init__(self, xtrain,ytrain= None, xtest=None, ytest=None):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest


    def Classification_start(self,string):
        #Description: the function decide which learner to apply and calculates all the metric for it.
        #INPUT: - string: name of the learner to apply
        #OUTPUT: -self.Accuracy, self.F1_score, self.BER: metrics to evaluate the performance of the pipeline
        print 'Starting Learning step'
        print   '----------------------'
        if string == 'SVC':
            self.SVC()
        elif string == 'Random_Forest':
            self.Random_forest()
        elif string == 'kNN':
            self.kNN()
        elif string == 'kMeans':
            self.kMeans_supervised()
        elif string == 'HierarchicalClustering':
            self.AggClustering()
        self.Accuracy_evaluation()
        self.F1_score_evaluation()
        self.BER_evaluation()
        return self.Accuracy, self.F1_score, self.BER

    def SVC(self):
        #Description: the function perform the Support Vector Classifier with linear Kernel.
        #            The train and test data are parameters of the class itself
        #INPUT:
        #OUTPUT:
        clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, \
         verbose=False, max_iter=10000)
        clf.fit(self.xtrain, self.ytrain)
        self.Prediction = clf.predict(self.xtest)

    def kNN(self):
        #Description: the function perform k - Nearest Neighbors with neighbors = 3 and,
        #            since the dataset is small, comparing all the points in a brute force approach.
        #            The train and test data are parameters of the class itself
        #INPUT:
        #OUTPUT:
        clf = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='brute', \
             p=2, metric='minkowski')
             #p=2 and minkowski is the euclidean distance, brute force will compute distance for each point.
        clf.fit(self.xtrain, self.ytrain)
        self.Prediction = clf.predict(self.xtest)

    def Random_forest(self):
        #Description: the function perform the Random Forset with 50 trees, 8 as max depth
        #            and 20 maximum leaves, to avoid overfitting on the training data.
        #            The train and test data are parameters of the class itself
        #INPUT:
        #OUTPUT:
        clf = RandomForestClassifier(n_estimators= 50, max_depth=8, max_leaf_nodes=20,random_state=0)
        clf.fit(self.xtrain, self.ytrain)
        self.Prediction = clf.predict(self.xtest)

    def kMeans_supervised(self):
        #Description: the function perform kMeans with k-means++ initialization to reduce convergence time.
        #            By now the number of cluster is 5, as the number of classes.
        #INPUT:
        #OUTPUT:
        NCLUSTER = 5
        kmeans = KMeans(init='k-means++',n_clusters=NCLUSTER, random_state=0)
        kmeans.fit(self.xtrain)
        label_convert = []
        for i in range(NCLUSTER):
            classes= []
            index=kmeans.labels_==i
            index_int = []
            for ind in range(index.size):
                if index[ind] ==True:
                    index_int.append(ind)
            for j in range(5):
                classes.append(sum(self.ytest[index_int]==j+1))
            max_label = max(classes)
            first = 1
            for j in range(5):
                if classes[j]==max_label and first == 1:
                    label_convert.append(j+1)
                    first = 0
        self.Prediction = kmeans.labels_
        for i in range(self.Prediction.size):
            self.Prediction[i] = label_convert[self.Prediction[i]]

    def AggClustering(self):
        #Description: the function perform Hierarchical Clustering
        #            By now the number of cluster is 5, as the number of classes.
        #INPUT:
        #OUTPUT:
        NCLUSTER = 5
        clt = AgglomerativeClustering(n_clusters=NCLUSTER, affinity = 'euclidean',linkage = 'ward')
        clt.fit(self.xtrain)
        label_convert = []
        for i in range(NCLUSTER):
            classes= []
            index=clt.labels_==i
            index_int = []
            for ind in range(index.size):
                if index[ind] ==True:
                    index_int.append(ind)
            for j in range(5):
                classes.append(sum(self.ytest[index_int]==j+1))
            max_label = max(classes)
            first = 1
            for j in range(5):
                if classes[j]==max_label and first == 1:
                    label_convert.append(j+1)
                    first = 0
        self.Prediction = clt.labels_
        for i in range(self.Prediction.size):
            self.Prediction[i] = label_convert[self.Prediction[i]]



    def Accuracy_evaluation(self):
        #Description: evaluate the accuracy of the learner.
        #INPUT:
        #OUTPUT:
        self.Accuracy = accuracy_score(self.ytest, self.Prediction)

    def F1_score_evaluation(self):
            #Description: evaluate the F1_score as mean of the F1_score of all the classes.
            #INPUT:
            #OUTPUT:
        self.F1_score = f1_score(self.ytest, self.Prediction, average='weighted') #other options are micro and macro, see documentatin

    def BER_evaluation(self):
            #Description: evaluate the BER as mean of the BER of all the classes.
            #INPUT:
            #OUTPUT:
        score =[]
        for label in [1,2,3,4,5]:
            class_predicted_well = 0
            class_true = self.ytest==label
            class_predicted =  self.Prediction==label
            for i in range(self.ytest.size):
                if class_true[i] == True and class_predicted[i] == True:
                    class_predicted_well = class_predicted_well + 1
            score.append(class_predicted_well/float(np.sum(class_true)))
        self.BER = 1 - np.mean(score)
        #we use an adapted version, that computes the errors on each class and make an average

    def Confusion_matrix(self):
        confusion_matrix = np.zeros([5,5])
        for i in range(self.Prediction.shape[0]):
            confusion_matrix[self.ytest[i].astype(int)-1,self.Prediction[i].astype(int)-1] += 1
        for i in range(5):
            print (confusion_matrix[i,i]/sum(confusion_matrix[i]))
