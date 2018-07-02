from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import GA
from sklearn import svm

class DimReduction():
    # initial model type, model pool and load in data_in,
    def __init__(self, mdl_type, dim_out, **para):
        # load para
        self.para = para
        self.model = []
        # define model type
        self.mdl_type = mdl_type
        self.dim_out = dim_out
    def Dataset_reduction(self, X_train, supervised = True, X_test = None, labels = None):
        #Description: based on the self.mdl_type it runs one of the 3 algorithms for the dimension reduction.
        #INPUT:  - X_train:train Dataset
        #        - supervised: if the method is supervised, we train the trasformation on the train dataset and trasform the test setself.
        #                        Otherwise, we only apply on train dataset.
        #        - X_test: test Dataset
        #        - labels: labels of the dataset, needed for GA and LDA.
        #OUTPUT: - X_tr: trasformed training set.
        #        - X_te: trasformed test set.
        if self.mdl_type == "GA":
            print '\nFeature Selection'
            print   '-----------------'
            params = {'C': 0.1, 'tol': 0.01, 'kernel' : 'linear', 'max_iter' : 10000, 'verbose' : False}
            GenAlgo = GA.GeneticAlgSelect( X_train, labels,svm.SVC, params, dim_out = self.dim_out)
            GenAlgo._perform_iter()
            GAopt = GenAlgo.mdl_pool[GenAlgo.elite_list[-1]]
            X_tr = X_train[:,GAopt.gene.astype(bool)]
            if supervised == True:
                X_te = X_test[:,GAopt.gene.astype(bool)]
        else:
            print '\nFeature Reduction'
            print   '-----------------'
            self.fit(X_train, labels)
            X_tr = self.transform(X_train)
            if supervised == True:
                X_te = self.transform(X_test)
        if supervised == True:
            return X_tr, X_te
        else:
            return X_tr

    def __PCA(self, data_in, to_fit):
        #Apply the PCA to the dataset
        X = preprocessing.scale((data_in-data_in.mean())/data_in.mean())
        if to_fit == 1:
            self.model = PCA(n_components=self.dim_out)
            self.model.fit(X)
        else :
            return self.model.transform(X)

    def __LDA(self,  data_in, to_fit, labels = None ):
        # Create an LDA that will reduce the data down to N feature
        # run an LDA and use it to transform the features
        if to_fit == 1:
            self.model = LinearDiscriminantAnalysis(n_components=self.dim_out)
            self.model.fit( data_in, labels)
        else :
            return self.model.transform(data_in)

    def fit(self, data_in, labels = None):
        if self.mdl_type == "PCA" :
            self.__PCA( data_in, 1)
        elif self.mdl_type == "LDA" :
            self.__LDA( data_in, 1, labels)

    def transform(self, data_in):
        if self.mdl_type == "PCA" :
            X_rdc = self.__PCA( data_in, 0)
        elif self.mdl_type == "LDA" :
            X_rdc = self.__LDA( data_in, 0)
        return X_rdc
