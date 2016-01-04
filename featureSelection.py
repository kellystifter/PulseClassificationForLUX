import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
import operator
from sklearn.ensemble import RandomForestClassifier


def RFbestfeat(clf,features):
    """
    Returns a list of the feature importance for the given random forest classifier

    Parameters:
    -----------
    clf : random forest classifier

    features : list
        List of features that are being used in the classifier
    """
    feat_importance = [(i,clf.feature_importances_[i]) for i in range(42)]
    label_importance = [(features[i[0]],i[1]) for i in feat_importance]
    label_importance.sort(key=operator.itemgetter(1))
    return label_importance


def featureSelection_cK(title,clf,X,y,CV,n_jobs=1,kvalues=[i for i in range(1,43)]):
    """
    Perform "Choose K best" feature selection and returns plot of performance vs. number of features

    Parameters
     ----------
    clf : object type that implements the "fit" and "predict" methods
         An object of that type which is cloned for each validation.

     title : string
         Title for the chart.

     X : array-like, shape (n_samples, n_features)
         Training vector, where n_samples is the number of samples and
         n_features is the number of features.

     y : array-like, shape (n_samples) or (n_samples, n_features), optional
         Target relative to X for classification or regression;
         None for unsupervised learning.

     cv : integer, cross-validation generator, optional
         If an integer is passed, it is the number of folds (defaults to 3).
         Specific cross-validation objects can be passed, see
         sklearn.cross_validation module for the list of possible objects

     n_jobs : integer, optional
         Number of jobs to run in parallel (default 1).

     kvalues : list
         Specify the number of features to test in each iteration
    """

    # Combine a feature-selection transform and a classifier to create a full-blown estimator
    transform = feature_selection.SelectKBest(feature_selection.f_classif)
    clf_k = Pipeline([('anova', transform), ('svc', clf)])

    # Plot the cross-validation score as a function of number of features
    score_means = list()
    score_stds = list()

    for k in kvalues:
        clf_k.set_params(anova__k=k)
        # Compute cross-validation score
        this_scores = cross_validation.cross_val_score(clf_k, X, y, cv=CV, n_jobs=1)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    transform.fit(X,y)
    feat_scores = transform.scores_

    plt.errorbar(kvalues, score_means, np.array(score_stds))

    plt.title(title)
    plt.xlabel('Number of Features')
    plt.ylabel('Prediction rate')

    plt.axis('tight')
    return plt,score_means,feat_scores
