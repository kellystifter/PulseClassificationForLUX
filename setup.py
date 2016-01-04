import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation

#create dataframe from csv file
designMatrix = pd.read_csv('designMatrix.csv')

designMatrix.loc[designMatrix.user_classification == 99, 'user_classification'] = 11
#drop the pulse that are labeled 'I don't know'
designMatrix = designMatrix[designMatrix.user_classification < 11]
userclf = designMatrix['user_classification']
luxclf = designMatrix['pulse_classification']

#remove columns that are sometimes infinity, NaN, or alway the same
dropLabels = ['user_classification','pulse_classification','gaus_fit_chisq',
              'exp_fit_chisq','top_bottom_ratio','pulse_area_positive_phe',
              'pulse_area_negative_phe']
designMatrix = designMatrix.drop(dropLabels,axis=1)

features = designMatrix.columns
#create the design matrix
dm = np.array(designMatrix)
#scale the data so each feature has mean 0 and stddev 1
X = preprocessing.scale(dm)

#function to transform a user classification to a LUX classification scheme
def transformToLUX(user):
  result = user
  result[result > 4] = 5
  return result

#singles out a single number
def singleNum(cls,num):
  result = cls
  result[result != num] = False
  result[result == num] = True
  return result

#create label vector
y = np.array(luxclf)
#y = np.array(userclf) #can switch what you use for labels

#define cross-validation strategy
folds = 4
cv = cross_validation.StratifiedKFold(y, n_folds=folds, shuffle=True)

#create matrix with the first column being the full label vector, and subsequent columns being just class one, just class two, etc...
l = [singleNum(np.array(y),i) for i in range(1,11)]
y_all = [y,l[0],l[1],l[2],l[3]]
