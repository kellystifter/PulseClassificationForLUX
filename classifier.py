import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from setup import singleNum
from sklearn.naive_bayes import GaussianNB

class combined(BaseEstimator, ClassifierMixin):
    """Combined with Base: Takes a list of classifiers, the first being the base classifier and the remaining four being the expert classifiers."""
    def __init__(self,clf):
        self.clf=clf

    def fit(self, X, y):
        def singleNum(cls,num):
          result = cls
          result[result != num] = False
          result[result == num] = True
          return result

        l = [singleNum(np.array(y),i) for i in range(1,11)]
        y_all = [y,l[0],l[1],l[2],l[3]]
        [self.clf[i].fit(X,y_all[i]) for i in range(0,len(self.clf))]

        return self

    def predict(self, X):
        self.pred = [self.clf[i].predict(X) for i in range(0,len(self.clf))]

        #Combine to create classifier:
        pred_comb = np.array(self.pred[0])
        pred_comb[(self.pred[1] == 1) & (self.pred[2] != 1) & (self.pred[3] != 1) & (self.pred[4] != 1)] = 1
        pred_comb[(self.pred[1] != 1) & (self.pred[2] == 1) & (self.pred[3] != 1) & (self.pred[4] != 1)] = 2
        pred_comb[(self.pred[1] != 1) & (self.pred[2] != 1) & (self.pred[3] == 1) & (self.pred[4] != 1)] = 3
        pred_comb[(self.pred[1] != 1) & (self.pred[2] != 1) & (self.pred[3] != 1) & (self.pred[4] == 1)] = 4

        #pred_comb[(pred[1] == 1) & (pred[2] == 1) & (pred[3] != 1) & (pred[4] != 1)] = 1
        pred_comb[(self.pred[1] == 1) & (self.pred[2] != 1) & (self.pred[3] == 1) & (self.pred[4] != 1)] = 1
        #pred_comb[(pred[1] == 1) & (pred[2] != 1) & (pred[3] != 1) & (pred[4] == 1)] = 1
        #pred_comb[(pred[1] != 1) & (pred[2] == 1) & (pred[3] == 1) & (pred[4] != 1)] = 1
        pred_comb[(self.pred[1] != 1) & (self.pred[2] == 1) & (self.pred[3] != 1) & (self.pred[4] == 1)] = 2
        #pred_comb[(pred[1] != 1) & (pred[2] != 1) & (pred[3] == 1) & (pred[4] == 1)] = 1

        return pred_comb

class voting(BaseEstimator, ClassifierMixin):
    """Combined without Base: Takes a list of four classifiers to act as experts"""
    def __init__(self,clf):
        self.clf=clf

    def fit(self, X, y):
        def singleNum(cls,num):
          result = cls
          result[result != num] = False
          result[result == num] = True
          return result

        l = [singleNum(np.array(y),i) for i in range(1,11)]
        y_all = [l[0],l[1],l[2],l[3]]
        [self.clf[i].fit(X,y_all[i]) for i in range(0,len(self.clf))]

        return self

    def predict(self, X):
        self.pred = [0]
        self.pred += [self.clf[i].predict(X) for i in range(0,len(self.clf))]

        #Combine to create classifier:
        pred_comb = np.empty(len(self.pred[1]))
        pred_comb.fill(5)
        pred_comb[(self.pred[1] == 1) & (self.pred[2] != 1) & (self.pred[3] != 1) & (self.pred[4] != 1)] = 1
        pred_comb[(self.pred[1] != 1) & (self.pred[2] == 1) & (self.pred[3] != 1) & (self.pred[4] != 1)] = 2
        pred_comb[(self.pred[1] != 1) & (self.pred[2] != 1) & (self.pred[3] == 1) & (self.pred[4] != 1)] = 3
        pred_comb[(self.pred[1] != 1) & (self.pred[2] != 1) & (self.pred[3] != 1) & (self.pred[4] == 1)] = 4

        #pred_comb[(pred[1] == 1) & (pred[2] == 1) & (pred[3] != 1) & (pred[4] != 1)] = 1
        pred_comb[(self.pred[1] == 1) & (self.pred[2] != 1) & (self.pred[3] == 1) & (self.pred[4] != 1)] = 1
        #pred_comb[(pred[1] == 1) & (pred[2] != 1) & (pred[3] != 1) & (pred[4] == 1)] = 1
        #pred_comb[(pred[1] != 1) & (pred[2] == 1) & (pred[3] == 1) & (pred[4] != 1)] = 1
        pred_comb[(self.pred[1] != 1) & (self.pred[2] == 1) & (self.pred[3] != 1) & (self.pred[4] == 1)] = 2
        #pred_comb[(pred[1] != 1) & (pred[2] != 1) & (pred[3] == 1) & (pred[4] == 1)] = 1

        return pred_comb

#Define final classifier
clf_list = [RandomForestClassifier(n_estimators=27, max_features=11),
        RandomForestClassifier(n_estimators=24, max_features=10),
        RandomForestClassifier(n_estimators=25, max_features=9),
        RandomForestClassifier(n_estimators=22, max_features=10)]
clf = voting(clf_list,.5)

#Split training and testing sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=.2)
clf.fit(X_train,y_train)
#Predict labels
pred_comb = clf.predict(X_test)

#Compute score of full classifier
score = cross_validation.cross_val_score(clf,X,y,cv = cv)
#Compute score of each expert
ind_scores = [cross_validation.cross_val_score(clf_list[i],X,y_all[i],cv = cv) for i in range(0,len(clf_list))]
#Confusion matrix
cm = confusion_matrix(y_test,pred_comb)

#Compile probability of each class
prob = [0]
prob += [clf_list[i].predict_proba(X_test) for i in range(0,len(clf_list))]

#Create list of which pulses were classified incorrectly
inc_ind = [i for i in range(len(y_test)) if y_test[i]!=pred_comb[i]]
incorrect = [(y_test[i],pred_comb[i],clf.pred[1][i],prob[1][i][1],clf.pred[2][i],prob[2][i][1],clf.pred[3][i],prob[3][i][1],clf.pred[4][i],prob[4][i][1]) for i in inc_ind]

#Print diagnostics
print(classification_report(y_test, pred_comb,digits=5))

#Other classifiers I tried:
#Naive Bayes classifier; calculate score
clf_nb = GaussianNB()
score_nb = cross_validation.cross_val_score(clf_nb,X,y,cv = cv)
score_nb_mean = score_nb.mean()
score_nb_std = score_nb.std()

#SVM classifier; calculate score
clf_svm = svm.SVC()
score_svm = cross_validation.cross_val_score(clf_svm,X,y,cv = cv)
score_svm_mean = score_svm.mean()
score_svm_std = score_svm.std()

#Random Forest classifier; calculate score, predicted labels, confusion matrix
clf_rf = RandomForestClassifier()
score_rf = cross_validation.cross_val_score(clf_rf,X,y,cv = cv)
score_rf_mean = score_rf.mean()
score_rf_std = score_rf.std()
clf_rf.fit(X_train,y_train)
pred_rf = clf_rf.predict(X_test)
cm_rf = confusion_matrix(y_test,pred_rf)

#Combined classifier with random forest base; calculate score, predicted labels, confusion matrix
cwbrf_list = [RandomForestClassifier(),
        RandomForestClassifier(),
        RandomForestClassifier(),
        RandomForestClassifier(),
        RandomForestClassifier()]
clf_cwbrf = combined(cwbrf_list)
score_cwbrf = cross_validation.cross_val_score(clf_cwbrf,X,y,cv = cv)
score_cwbrf_mean = score_cwbrf.mean()
score_cwbrf_std = score_cwbrf.std()
clf_cwbrf.fit(X_train,y_train)
pred_cwbrf = clf_cwbrf.predict(X_test)
cm_cwbrf = confusion_matrix(y_test,pred_cwbrf)
ind_scores = [cross_validation.cross_val_score(cwbrf_list[i],X,y_all[i],cv = cv) for i in range(0,len(cwbrf_list))]

#Combined classifier with SVM base; calculate score, predicted labels, confusion matrix
cwbsvm_list = [svm.SVC(),
        RandomForestClassifier(),
        RandomForestClassifier(),
        RandomForestClassifier(),
        RandomForestClassifier()]
clf_cwbsvm = combined(cwbsvm_list)
score_cwbsvm = cross_validation.cross_val_score(clf_cwbsvm,X,y,cv = cv)
score_cwbsvm_mean = score_cwbsvm.mean()
score_cwbsvm_std = score_cwbsvm.std()
clf_cwbsvm.fit(X_train,y_train)
pred_cwbsvm = clf_cwbsvm.predict(X_test)
cm_cwbsvm = confusion_matrix(y_test,pred_cwbsvm)
