from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
import numpy as np
import pandas as pd


class stackingMetaClassifier:

    def __init__(self, baseModels, folds = 5):
        self.base_models = baseModels
        # self.stacking_model = stackingModel
        self.kf = KFold(n_splits=folds, shuffle=True)


    def fitbase_generateMetaTrain(self, train_X, train_Y, fname):
        self.splits = list(self.kf.split(train_X, train_Y))
        self.level2_train_X = np.zeros((len(train_Y), len(self.base_models)))

        for i, model in enumerate(self.base_models):
            print('training {}'.format(model[0]))
            self.level2_train_X[:, i] = self.__oof(model[1], train_X, train_Y)

        self.level2_train_X = pd.DataFrame(self.level2_train_X, columns=[x[0] for x in self.base_models])

        self.level2_train_X.to_csv(fname, index=False)
        self.train_Y = train_Y

    def fitMeta(self,stackingModel, df2append):
        self.stacking_model = stackingModel

        level2_train_X = pd.concat([df2append, self.level2_train_X], axis = 1)

        self.stacking_model.fit(level2_train_X, self.train_Y)
        # level2_test_X = pd.concat([test_X_50, pd.DataFrame(level2_test_X, columns=['m1','m2','m3', 'm4'])], axis=1)

        # return self.stacking_model.predict_proba(level2_test_X)

    def __oof(self, model, train_X, train_Y):
        train_result = np.zeros((len(train_Y),))

        for trainIndex, validIndex in self.splits:
            sample_train_X = train_X.iloc[trainIndex]
            sample_train_Y = train_Y.iloc[trainIndex]
            sample_valid_X = train_X.iloc[validIndex]
            model.fit(sample_train_X, sample_train_Y)
            train_result[validIndex] = model.predict_proba(sample_valid_X)[:,1]

        model.fit(train_X, train_Y)

        return train_result.ravel()

    def predict_proba(self,test_X, df2append, fname):
        level2_test_X = np.zeros((len(test_X), len(self.base_models)))
        for i, model in enumerate(self.base_models):
            level2_test_X[:, i] = model[1].predict_proba(test_X)[:,1]

        level2_test_X = pd.DataFrame(level2_test_X, columns=[x[0] for x in self.base_models])

        level2_test_X.to_csv(fname, index=False)

        level2_test_X = pd.concat([df2append, level2_test_X], axis=1)
        return self.stacking_model.predict_proba(level2_test_X)


    def auc_evaluate(self, test_X, test_Y,df2append, fname):
        for name, model in self.base_models:
            base_yhat = model.predict_proba(test_X)[:,1]
            print('Base Model {} auc: {}'.format(name, roc_auc_score(test_Y, base_yhat)))
        yhat = self.predict_proba(test_X, df2append, fname)[:,1]
        auc = roc_auc_score(test_Y, yhat)
        print("stacker: " + str(type(self.stacking_model)) + " auc: " + str(auc))
        return auc



def getPCA(n_comp, train_X, test_X):
    pca = PCA(n_components=n_comp)
    pca.fit(train_X)
    train_X_pca = pd.DataFrame(pca.transform(train_X))
    test_X_pca = pd.DataFrame(pca.transform(test_X))
    return train_X_pca, test_X_pca

def slct_features(num):
    extraTree = ExtraTreesClassifier(n_estimators=100, min_samples_split=11, n_jobs=-1)

    extraTree.fit(train_X, train_Y)

    importance = {}
    for i, imp in enumerate(extraTree.feature_importances_):
        importance[train_X.columns[i]] = imp
    features2use = []
    for f in sorted(importance.items(), key=lambda x: -x[1])[:50]:
        features2use.append(f[0])
    return features2use


if __name__ == '__main__':


    train = pd.read_csv("data/train_12_3.csv")
    valid = pd.read_csv("data/valid_12_3.csv")
    test = pd.read_csv("data/test_12_3.csv")
    train = train.append(valid)
    train = train.append(test)

    train.drop(columns='ID', inplace=True)
    train_X = train.drop(columns="TARGET")
    train_Y = train["TARGET"]

    scaler = StandardScaler()
    train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=train_X.columns)

    realtest_X = pd.read_csv('data/test_cleaned_12_9.csv')
    ID = realtest_X['ID']
    realtest_X.drop(columns='ID', inplace=True)
    realtest_X = pd.DataFrame(scaler.transform(realtest_X), columns=train_X.columns)


    train_X_pca, test_X_pca = getPCA(10, train_X, realtest_X)
    #
    # features2use = slct_features(50)
    #
    # train_X_2pc50 = train_X.loc[:, features2use]
    # test_X_2pc50 = realtest_X.loc[:, features2use]
    # for col in train_X_pca.columns:
    #     train_X_2pc50['PCA', col] = train_X_pca.loc[:, col]
    #     test_X_2pc50['PCA', col] = test_X_pca.loc[:, col]



    dtree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=200)
    # lda = LDA(shrinkage=0.01, solver='eigen')
    # qda = QDA(reg_param=0.6)
    xgb = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100)
    logreg = LogisticRegression(penalty='l1',C=0.5)
    mlp = mlp(hidden_layer_sizes=8, learning_rate_init=0.01, max_iter=2000)
    rf = RandomForestClassifier(max_depth=17, min_samples_leaf=10, n_estimators=800)
    knc = knn(n_neighbors=500, p=1)

    base_models = [['tree', dtree],['xgb', xgb], ['logreg',logreg], ['mlp',mlp], ['rf', rf], ['knn', knc]]
    stacker = XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.1)

    meta = stackingMetaClassifier(base_models)
    meta.fitbase_generateMetaTrain(train_X, train_Y, 'data/level2realtrain.csv')
    meta.fitMeta(stacker, train_X)
    result = meta.predict_proba(realtest_X, realtest_X, 'data/level2realtest.csv' )[:,1]
    print(stacker.feature_importances_)

    subm = pd.DataFrame({'ID':ID,'TARGET':result})
    subm.to_csv('submission.csv', index = False)







