import optunity
import optunity.metrics
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def methodSelection(data,labels):

    def train_svm(data, labels, kernel, C, gamma, degree, coef0):
        """A generic SVM training function, with arguments based on the chosen kernel."""
        if kernel == 'linear':
            model = SVC(kernel=kernel, C=C)
        elif kernel == 'poly':
            model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
        elif kernel == 'rbf':
            model = SVC(kernel=kernel, C=C, gamma=gamma)
        else:
            raise ArgumentError("Unknown kernel function: %s" % kernel)
        model.fit(data, labels)
        return model

    search = {'algorithm': {'k-nn': {'n_neighbors': [1, 10]},
                            'SVM': {'kernel': {'linear': {'C': [0, 2]},
                                               'rbf': {'gamma': [0, 1], 'C': [0, 10]},
                                               'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1]}
                                               }
                                    },
                            'naive-bayes': None,
                            'random-forest': {'n_estimators': [10, 30],
                                              'max_features': [5, 20]}
                            }
             }

    @optunity.cross_validated(x=data, y=labels, num_folds=4)
    def performance(x_train, y_train, x_test, y_test,
                    algorithm, n_neighbors=None, n_estimators=None, max_features=None,
                    kernel=None, C=None, gamma=None, degree=None, coef0=None):
        # fit the model
        if algorithm == 'k-nn':
            model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
            model.fit(x_train, y_train)
        elif algorithm == 'SVM':
            model = train_svm(x_train, y_train, kernel, C, gamma, degree, coef0)
        elif algorithm == 'naive-bayes':
            model = GaussianNB()
            model.fit(x_train, y_train)
        elif algorithm == 'random-forest':
            model = RandomForestClassifier(n_estimators=int(n_estimators),
                                           max_features=int(max_features))
            model.fit(x_train, y_train)
        else:
            raise ArgumentError('Unknown algorithm: %s' % algorithm)

        # predict the test set
        if algorithm == 'SVM':
            predictions = model.decision_function(x_test)
        else:
            predictions = model.predict_proba(x_test)[:, 1]

        return optunity.metrics.roc_auc(y_test, predictions, positive=True)

    optimal_configuration, info, _ = optunity.maximize_structured(performance,
                                                                  search_space=search,
                                                                  num_evals=300)

    solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
    print('Solution\n========')
    print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))
    print(info.optimum)
