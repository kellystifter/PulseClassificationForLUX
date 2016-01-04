# PulseClassificationForLUX
Machine Learning Project for CS229 which classifies pulses in LUX data

Description of files:

createDesignMatrix.py:
  To be run as a script on a PDSF computer. Pulls the necessary information from RQ files using the python RQ reader and creates a .csv file with the design matrix. Requires the .csv file containg the pLUX classififcation information to be in the same directory.
  
setup.py:
  To be run as a script in your local workspace. Imports the design matrix, sets up all necessary variables. Requires the .csv file containing the design matrix to be in the same directory.
  
PCA.py:
  To be run as a script in your local workspace after running setup.py. Performs principle components analysis and reduces the dimension of the data to 2.
  
classifier.py:
  To be run as a script in your local workspace after running setup.py. Defines 2 new classifiers. Defines a final classifier, as well a several others I tried. Computes the score, confusion matrix, etc. for many of them.

learningCurve.py:
  Defines functions for diagnostics to be used after defining a classifier. Plot learning curve or validation curve (NOT SURE IF VALIDATION CURVE WORKS).

featureSelection.py:
  Defines functions for diagnostics to be used after defining a classifier. Compute the feature importances for a random forest classifier, or perform 'Choose K best' feature selection (doesn't work with random forests).

methodSelection.py:
  Uses a package called Optunity. It was kind of a pain to install, and finicky to work with. I copied the function from their website. It searches through parameter space and tells you which classifier and which hyperparameters are best.
