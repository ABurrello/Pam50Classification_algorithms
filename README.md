# Pam50 Classification of Breast Cancer. 
## An overview of different preprocessing thecniques, imbalance management and learning algorithms.

This repository provides the PYTHON codes used to run the experiments. We used Python 2.7.14.

## Prerequisites

* Running version of python 2.7.
* Packages to be installed: sklearn (scikit-learn), scipy, numpy, pandas, imblearn. You can install by e.g.
> `pip install pandas`

or

> `python -m pip install pandas`

or 

> `conda install pandas`



## Dataset
The dataset used could be downloaded from [here](https://www.dropbox.com/s/g2sggr622t6agu4/Dataset.mat?dl=0)

## Files:
The starting file is the main, while all the others are the classes used inside:
* **Main.py**:	manages the k-fold training/testing loop and calls all the classes method
* **GA.py**:   Genetic Algortihm Class for feature selection. 
* **Imbalance_manager.py**: runs the selected method for imbalance issue
* **DR.py**: runs the selected method for dimensionality reduction
* **Classification.py**: runs the selected method for classification and Scores computation
* **Opening_design.py**: manages user interface and defines the pipeline 

## Running the tests 

To achieve the results, you can run the script in two different ways:
* by normally call the script with 
> `python .\Main.py`

and following the I/O user interface selecting the preferred methods. 
**N.B.** If the *Unsupervised* pipeline is choosen, the feature reduction and class imbalance management are fixed, because we force methods that don't need any labels.

* using the command line through the parameter --fast True. The default pipeline in this case is : PCA, no class balancing,SVC.
With the command 
> `python .\Main.py -h`

you can see the parameter to set and how to fill them.
Each part of the pipeline can be modified by inserting the corresponding arguments (see --help).
Example:

> `python .\Main.py --fast True --reduction LDA --imbalance SMOTE --supervised Random_Forest`

The implemented methods are:

* **Supervised**: 		SVC, KNN, RandomForest
* **Unsupervised**: 		Kmeans, Hierarchical clustering
* **Unbalanced method**: 	SMOTE, SMOTE + ENN , RandomOverSampling
* **Dim Reduction**: 		PCA, LDA, GA

## Notes on the algorithms
* The algorithm with the supervised pipeline use 10-fold cross validation for test.
* The unsupervised pipeline is forced, you can change only the unsupervised method: in addition we use the full dataset in the unsupervised version.
* The Dimensional reduction methods produce 600 features by now. We will fine tune this parameter. 
* You can change number of folds and Number of features in the algorithm changing *NFOLDS* and *Nfeatures*. 
* The performance is evaluated through *Accuracy*, *F1 score* and *Balanced Error rate* (this last one is better if lower and takes into account the unbalancing of the classes).

