import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as rf
# from lightgbm import LGBMModel
from xgboost import XGBClassifier as xgb

# Load Training and Validation Datasets
train = pd.read_csv("data/train_12_3.csv")
valid = pd.read_csv("data/valid_12_3.csv")

train = train.append(valid)
train.drop(columns='ID', inplace=True)
train_X = train.drop(columns="TARGET")
train_Y = train["TARGET"]
# print(train_X.head())

scaler = StandardScaler()
train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=train_X.columns)
print(train.shape)

test = pd.read_csv("data/test_12_3.csv")
test.drop(columns='ID', inplace=True)
test_X = test.drop(columns="TARGET")
test_Y = test["TARGET"]
test_X = pd.DataFrame(scaler.transform(test_X), columns=train_X.columns)

models2consider = [['DecisionTree', DecisionTreeClassifier(), {'max_depth': [3, 5, 7, 9, 11, 13, 15],
                                                               'min_samples_leaf': [30, 40, 100, 200, 500, 1000,
                                                                                    2000]}],
                   ['kNN', KNC(), {'n_neighbors': [1, 3, 5, 10, 20, 50, 100, 500], 'p': [1, 2, 3]}],
                   ['LDA', LDA(), {'shrinkage': ['auto', 0.01, 0.1, 0.3, 0.6], 'solver': ['lsqr', 'eigen']}],
                   ['QDA', QDA(), {'reg_param': [0.001, 0.01, 0.1, 0.3, 0.6]}],
                   ['SVM', SVC(probability=True, gamma='scale', kernel='poly'),
                    {'tol': [0.0001, 0.001, 0.01, 0.1], 'C': [0.01, 0.1, 1, 10, 100, 1000]}],
                   ['logistic', LogisticRegression(solver='saga', max_iter=-1),
                    {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 0.5, 1, 10]}],
                   ['MLP', MLP(max_iter=2000),
                    {'activation': ['tanh', 'relu'], 'hidden_layer_sizes': [8, 16, 32, [32, 16], [16, 8], [16, 8, 4]],
                     'learning_rate_init': [0.001, 0.01, 0.1]}],
                   ['XGB', xgb(), {'max_depth': [3, 7, 9, 15], 'n_estimators': [10, 30, 50, 100, 300],
                                   'learning_rate': [0.001, 0.01, 0.1]}],
                   # ['LightGBM', LGBMModel(),
                   #  {'max_depth': [3, 5, 7, 9], 'n_estimators': [50, 100, 300], 'learning_rate': [0.001, 0.01, 0.1],
                   #   'num_leaves': [10, 20, 200, 500, 1000, 2000]}],
                   ['RF', rf(), {'n_estimators': [100, 500, 800, 1000], 'max_depth': [3, 7, 11, 13, 15, 17],
                                 'min_samples_leaf': [10, 20, 200, 500, 1000, 2000]}]]

allmodel_resu = []
for i in range(len(models2consider)):
    model_name = models2consider[i][0]
    clf = models2consider[i][1]
    parameters = models2consider[i][2]

    print(model_name)

    model_resu = {'name': model_name}
    CVresu = GridSearchCV(clf, parameters, cv=3, scoring='roc_auc', return_train_score=False, n_jobs=-1)
    CVresu = CVresu.fit(train_X, train_Y)
    best = CVresu.best_estimator_
    model_resu['best_param'] = CVresu.best_params_
    model_resu['best_CV_AUC'] = CVresu.best_score_
    yhat = best.predict_proba(test_X)[:, 1]
    print((yhat >= 0.5).sum())
    model_resu['test_AUC'] = roc_auc_score(test_Y, yhat)
    print(model_resu)
    allmodel_resu.append(model_resu)
