Requirements:
python2.7
packages:  sklearn, scipy, numpy, pandas, imblearn

Files:
-Main.py				:	manages the k-fold training/testing loop and calls all the classes method
-GA.py					:   Genetic Algortihm Class for feature selection. 
-Imbalance_manager.py   :   runs the selected method for imbalance issue
-DR.py					: 	runs the selected method for dimensionality reduction
-Classification.py		:	runs the selected method for classification and Scores computation
-Opening_design.py		:	manages user interface and defines the pipeline 

The pipeline can be chose in two way:
-by following I/O user interface
-by command line using --fast True argument 
   the default one is : PCA - no class balancing - SVC
   each part of the pipeline can be modified by inserting the corresponding arguments (see --help)
   
   e.g. python main.py --fast True --reduction LDA --imbalance SMOTE --supervised Random_Forest

The implemented methods are:

Supervised: 		SVC, KNN, RandomForest
Unsupervised : 		Kmeans, Hierarchical clustering
Unbalanced method: 	SMOTE, SMOTE + ENN , RandomOverSampling
Dim Reduction: 		PCA, LDA, GA

The algorithm use 10-fold cross validation for test.
The DR methods produce 600 features. 
You can change these two constant at the beginning of main.py 

For each fold  Accuracy, F1score, BER are showed.

