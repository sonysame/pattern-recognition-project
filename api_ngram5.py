import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
import xgboost as xgb
import numpy as np
from sklearn.svm import SVC
import pandas as pd
import os
import time
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import learning_curve

def _2gram_cross_validation(t):
    print("#################2gram_cross_validation_result#################\n")

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    total_score_result=[]

    _2gram_selected_df = pd.read_pickle("./2gram_selected.pkl")
    for feature in tqdm(t, mininterval=1):
        score_result = []
        X_train = _2gram_selected_df.iloc[:-1, :feature]
        y_train = _2gram_selected_df.iloc[:-1, -1]
        rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=300, max_depth=6, min_samples_leaf=16,
                                        min_samples_split=8, random_state=0)
        gnb = GaussianNB()
        knn = KNeighborsClassifier(n_neighbors=8)
        dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
        gb = GradientBoostingClassifier(random_state=0)

        score_result.append(np.mean(cross_val_score(rf_clf, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(gnb, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(knn, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(dtc, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(gb, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))

        total_score_result.append(score_result)

    total_score_result=np.array(total_score_result)

    print("##cross_validation score#########################################\n")
    print(total_score_result)

    #fig=plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(t, total_score_result[:, 0], color='k', label='RF')
    ax.plot(t, total_score_result[:, 1], color='b', label='NB')
    ax.plot(t, total_score_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_score_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_score_result[:, 4], color='c', label='GB')
    #ax.plot(t, total_accuracy_result[:, 5], color='m', label='SVM')
    ax.set_title("2-gram cross validation score")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$accuracy$")
    ax.legend(loc='upper left')


def _3gram_cross_validation(t):
    print("#################3gram_cross_validation_result#################\n")

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    total_score_result = []

    _3gram_selected_df = pd.read_pickle("./3gram_selected.pkl")
    for feature in tqdm(t, mininterval=1):
        score_result = []
        X_train = _3gram_selected_df.iloc[:-1, :feature]
        y_train = _3gram_selected_df.iloc[:-1, -1]
        rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=300, max_depth=6, min_samples_leaf=16,
                                        min_samples_split=8, random_state=0)
        gnb = GaussianNB()
        knn = KNeighborsClassifier(n_neighbors=8)
        dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
        gb = GradientBoostingClassifier(random_state=0)

        score_result.append(np.mean(cross_val_score(rf_clf, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(gnb, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(knn, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(dtc, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(gb, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))

        # print('n_splits={}, cross validation score: {}'.format(n, scores))
        total_score_result.append(score_result)

    total_score_result = np.array(total_score_result)

    print("##cross_validation score#########################################\n")
    print(total_score_result)

    # fig=plt.figure()
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(t, total_score_result[:, 0], color='k', label='RF')
    ax.plot(t, total_score_result[:, 1], color='b', label='NB')
    ax.plot(t, total_score_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_score_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_score_result[:, 4], color='c', label='GB')
    # ax.plot(t, total_accuracy_result[:, 5], color='m', label='SVM')
    ax.set_title("3-gram cross validation score")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$accuracy$")
    ax.legend(loc='upper left')


def _4gram_cross_validation(t):
    print("#################4gram_cross_validation_result#################\n")

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    total_score_result = []

    _4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
    for feature in tqdm(t, mininterval=1):
        score_result = []
        X_train = _4gram_selected_df.iloc[:-1, :feature]
        y_train = _4gram_selected_df.iloc[:-1, -1]
        rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=300, max_depth=6, min_samples_leaf=16,
                                        min_samples_split=8, random_state=0)
        gnb = GaussianNB()
        knn = KNeighborsClassifier(n_neighbors=8)
        dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
        gb = GradientBoostingClassifier(random_state=0)

        score_result.append(np.mean(cross_val_score(rf_clf, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(gnb, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(knn, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(dtc, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))
        score_result.append(np.mean(cross_val_score(gb, X_train, y_train, scoring="accuracy", cv=kfold, n_jobs=-1)))

        # print('n_splits={}, cross validation score: {}'.format(n, scores))
        total_score_result.append(score_result)

    total_score_result = np.array(total_score_result)

    print("##cross-validation score#########################################\n")
    print(total_score_result)

    # fig=plt.figure()
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t, total_score_result[:, 0], color='k', label='RF')
    ax.plot(t, total_score_result[:, 1], color='b', label='NB')
    ax.plot(t, total_score_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_score_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_score_result[:, 4], color='c', label='GB')
    # ax.plot(t, total_accuracy_result[:, 5], color='m', label='SVM')
    ax.set_title("4-gram cross validation score")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$accuracy$")
    ax.legend(loc='upper left')

def _2gram_t(t):
    print("#################2gram_result#################\n")
    _2gram_selected_df = pd.read_pickle("./2gram_selected.pkl")
    _2gram_selected_test_df = pd.read_pickle("./2gram_selected_test.pkl")
    total_time_result=[]
    total_accuracy_result = []
    total_precision_result = []
    total_f1_result = []
    for feature in tqdm(t, mininterval=1):
        time_result = []
        accuracy_result = []
        precision_result = []
        f1_result = []
        X_train = _2gram_selected_df.iloc[:-1, :feature]
        X_test = _2gram_selected_test_df.iloc[:-1, :feature]
        y_train = _2gram_selected_df.iloc[:-1, -1]
        y_test = _2gram_selected_test_df.iloc[:-1, -1]

        params = {
            'n_estimators': [300, 400, 500],
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [8, 12, 16],
            'min_samples_split': [8, 12, 16]
        }

        # rf_clf=RandomForestClassifier(random_state=0, n_jobs=-1)
        # grid_cv=GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
        # grid_cv.fit(X_train, y_train)

        # print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
        # print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))
        start_time=time.time()
        rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=300, max_depth=6, min_samples_leaf=16,
                                        min_samples_split=8, random_state=0, n_jobs=-1)
        rf_clf.fit(X_train, y_train)
        pred = rf_clf.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred, average='weighted', zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time()-start_time)
        # print("Random Forest 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        start_time=time.time()
        gnb = GaussianNB()
        pred = gnb.fit(X_train, y_train).predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        # print("GaussianNB 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        """
        knn_result=[]
        for k in range(1,20):
            knn= KNeighborsClassifier(n_neighbors = k)
            knn.fit(X_train, y_train)
            pred=knn.predict(X_test)
            knn_result.append(accuracy_score(y_test,pred))
            #print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
        plt.plot(range(1,20), knn_result)
        plt.title("Best k for KNN")
        plt.xlabel("$k$")
        plt.ylabel("$accuracy$")
        plt.show()
        """
        start_time=time.time()
        knn = KNeighborsClassifier(n_neighbors=8, n_jobs=-1)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        #knn_result.append(accuracy_score(y_test, pred))
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time()-start_time)
        # print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        start_time=time.time()
        dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
        dtc.fit(X_train, y_train)
        pred = dtc.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        # print("Decision Tree Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        start_time=time.time()
        gb = GradientBoostingClassifier(random_state=0)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        # print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
        """
        start_time=time.time()
        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        #print("SVM 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))
        """
        total_time_result.append(time_result)
        total_accuracy_result.append(accuracy_result)
        total_precision_result.append(precision_result)
        total_f1_result.append(f1_result)

    total_time_result=np.array(total_time_result)
    total_accuracy_result = np.array(total_accuracy_result)
    total_precision_result = np.array(total_precision_result)
    total_f1_result=np.array(total_f1_result)

    print("##time#########################################\n")
    print(total_time_result)
    print("##accuracy#####################################\n")
    print(total_accuracy_result)
    print("##precision####################################\n")
    print(total_precision_result)
    print("##f1#######################################\n")
    print(total_f1_result)


    #fig=plt.figure()
    ax = fig.add_subplot(3, 3, 1)
    ax.plot(t, total_accuracy_result[:, 0], color='k', label='RF')
    ax.plot(t, total_accuracy_result[:, 1], color='b', label='NB')
    ax.plot(t, total_accuracy_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_accuracy_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_accuracy_result[:, 4], color='c', label='GB')
    #ax.plot(t, total_accuracy_result[:, 5], color='m', label='SVM')
    ax.set_title("2-gram accuracy")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$accuracy$")
    ax.legend(loc='upper left')

    ax = fig.add_subplot(3, 3, 2)
    ax.plot(t, total_precision_result[:, 0], color='k', label='RF')
    ax.plot(t, total_precision_result[:, 1], color='b', label='NB')
    ax.plot(t, total_precision_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_precision_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_precision_result[:, 4], color='c', label='GB')
    #ax.plot(t, total_precision_result[:, 5], color='m', label='SVM')
    ax.set_title("2-gram precision")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$precision$")
    ax.legend(loc='upper left')

    ax = fig.add_subplot(3, 3, 3)
    ax.plot(t, total_f1_result[:, 0], color='k', label='RF')
    ax.plot(t, total_f1_result[:, 1], color='b', label='NB')
    ax.plot(t, total_f1_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_f1_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_f1_result[:, 4], color='c', label='GB')
    #ax.plot(t, total_f1_result[:, 5], color='m', label='SVM')
    ax.set_title("2-gram f1-score")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$f1$")
    ax.legend(loc='upper left')


def _3gram_t(t):
    print("#################3gram_result#################\n")

    _3gram_selected_df = pd.read_pickle("./3gram_selected.pkl")
    _3gram_selected_test_df = pd.read_pickle("./3gram_selected_test.pkl")
    total_time_result=[]
    total_accuracy_result = []
    total_precision_result = []
    total_f1_result = []
    for feature in tqdm(t, mininterval=1):
        time_result = []
        accuracy_result = []
        precision_result = []
        f1_result = []
        X_train = _3gram_selected_df.iloc[:-1, :feature]
        X_test = _3gram_selected_test_df.iloc[:-1, :feature]
        y_train = _3gram_selected_df.iloc[:-1, -1]
        y_test = _3gram_selected_test_df.iloc[:-1, -1]

        params = {
            'n_estimators': [300, 400, 500],
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [8, 12, 16],
            'min_samples_split': [8, 12, 16]
        }

        # rf_clf=RandomForestClassifier(random_state=0, n_jobs=-1)
        # grid_cv=GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
        # grid_cv.fit(X_train, y_train)

        # print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
        # print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))
        start_time=time.time()
        rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=300, max_depth=6, min_samples_leaf=16,
                                        min_samples_split=8, random_state=0, n_jobs=-1)
        rf_clf.fit(X_train, y_train)
        pred = rf_clf.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred, average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time()-start_time)
        # print("Random Forest 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        start_time=time.time()
        gnb = GaussianNB()
        pred = gnb.fit(X_train, y_train).predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        # print("GaussianNB 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        """
        knn_result=[]
        for k in range(1,20):
            knn= KNeighborsClassifier(n_neighbors = k)
            knn.fit(X_train, y_train)
            pred=knn.predict(X_test)
            knn_result.append(accuracy_score(y_test,pred))
            #print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
        plt.plot(range(1,20), knn_result)
        plt.title("Best k for KNN")
        plt.xlabel("$k$")
        plt.ylabel("$accuracy$")
        plt.show()
        """
        start_time=time.time()
        knn = KNeighborsClassifier(n_neighbors=8, n_jobs=-1)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        #knn_result.append(accuracy_score(y_test, pred))
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time()-start_time)
        # print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        start_time=time.time()
        dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
        dtc.fit(X_train, y_train)
        pred = dtc.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        # print("Decision Tree Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        start_time=time.time()
        gb = GradientBoostingClassifier(random_state=0)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        # print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
        """
        start_time=time.time()
        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        #print("SVM 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))
        """
        total_time_result.append(time_result)
        total_accuracy_result.append(accuracy_result)
        total_precision_result.append(precision_result)
        total_f1_result.append(f1_result)

    total_time_result=np.array(total_time_result)
    total_accuracy_result = np.array(total_accuracy_result)
    total_precision_result = np.array(total_precision_result)
    total_f1_result=np.array(total_f1_result)

    print("##time#########################################\n")
    print(total_time_result)
    print("##accuracy#####################################\n")
    print(total_accuracy_result)
    print("##precision####################################\n")
    print(total_precision_result)
    print("##f1#######################################\n")
    print(total_f1_result)


    #fig=plt.figure()
    ax = fig.add_subplot(3, 3, 4)
    ax.plot(t, total_accuracy_result[:, 0], color='k', label='RF')
    ax.plot(t, total_accuracy_result[:, 1], color='b', label='NB')
    ax.plot(t, total_accuracy_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_accuracy_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_accuracy_result[:, 4], color='c', label='GB')
    #ax.plot(t, total_accuracy_result[:, 5], color='m', label='SVM')
    ax.set_title("3-gram accuracy")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$accuracy$")
    ax.legend(loc='upper left')

    ax = fig.add_subplot(3, 3, 5)
    ax.plot(t, total_precision_result[:, 0], color='k', label='RF')
    ax.plot(t, total_precision_result[:, 1], color='b', label='NB')
    ax.plot(t, total_precision_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_precision_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_precision_result[:, 4], color='c', label='GB')
    #ax.plot(t, total_precision_result[:, 5], color='m', label='SVM')
    ax.set_title("3-gram precision")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$precision$")
    ax.legend(loc='upper left')

    ax = fig.add_subplot(3, 3, 6)
    ax.plot(t, total_f1_result[:, 0], color='k', label='RF')
    ax.plot(t, total_f1_result[:, 1], color='b', label='NB')
    ax.plot(t, total_f1_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_f1_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_f1_result[:, 4], color='c', label='GB')
    #ax.plot(t, total_f1_result[:, 5], color='m', label='SVM')
    ax.set_title("3-gram f1-score")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$f1$")
    ax.legend(loc='upper left')


def _4gram_t(t):
    print("#################4gram_result#################\n")

    _4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
    _4gram_selected_test_df = pd.read_pickle("./4gram_selected_test.pkl")
    total_time_result=[]
    total_accuracy_result = []
    total_precision_result = []
    total_f1_result = []
    for feature in tqdm(t, mininterval=1):
        time_result = []
        accuracy_result = []
        precision_result = []
        f1_result = []
        X_train = _4gram_selected_df.iloc[:-1, :feature]
        X_test = _4gram_selected_test_df.iloc[:-1, :feature]
        y_train = _4gram_selected_df.iloc[:-1, -1]
        y_test = _4gram_selected_test_df.iloc[:-1, -1]

        params = {
            'n_estimators': [300, 400, 500],
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [8, 12, 16],
            'min_samples_split': [8, 12, 16]
        }

        # rf_clf=RandomForestClassifier(random_state=0, n_jobs=-1)
        # grid_cv=GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
        # grid_cv.fit(X_train, y_train)

        # print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
        # print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))
        start_time=time.time()
        rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=300, max_depth=6, min_samples_leaf=16,
                                        min_samples_split=8, random_state=0, n_jobs=-1)
        rf_clf.fit(X_train, y_train)
        pred = rf_clf.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred, average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time()-start_time)
        # print("Random Forest 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        start_time=time.time()
        gnb = GaussianNB()
        pred = gnb.fit(X_train, y_train).predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        # print("GaussianNB 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        """
        knn_result=[]
        for k in range(1,20):
            knn= KNeighborsClassifier(n_neighbors = k)
            knn.fit(X_train, y_train)
            pred=knn.predict(X_test)
            knn_result.append(accuracy_score(y_test,pred))
            #print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
        plt.plot(range(1,20), knn_result)
        plt.title("Best k for KNN")
        plt.xlabel("$k$")
        plt.ylabel("$accuracy$")
        plt.show()
        """
        start_time=time.time()
        knn = KNeighborsClassifier(n_neighbors=8, n_jobs=-1)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        #knn_result.append(accuracy_score(y_test, pred))
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time()-start_time)
        # print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        start_time=time.time()
        dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
        dtc.fit(X_train, y_train)
        pred = dtc.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        # print("Decision Tree Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        start_time=time.time()
        gb = GradientBoostingClassifier(random_state=0)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        # print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
        """
        start_time=time.time()
        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        precision_result.append(precision_score(y_test, pred,average='weighted',zero_division=0))
        f1_result.append(f1_score(y_test, pred,average='weighted'))
        time_result.append(time.time() - start_time)
        #print("SVM 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))
        """
        total_time_result.append(time_result)
        total_accuracy_result.append(accuracy_result)
        total_precision_result.append(precision_result)
        total_f1_result.append(f1_result)

    total_time_result=np.array(total_time_result)
    total_accuracy_result = np.array(total_accuracy_result)
    total_precision_result = np.array(total_precision_result)
    total_f1_result=np.array(total_f1_result)

    print("##time#########################################\n")
    print(total_time_result)
    print("##accuracy#####################################\n")
    print(total_accuracy_result)
    print("##precision####################################\n")
    print(total_precision_result)
    print("##f1#######################################\n")
    print(total_f1_result)


    #fig=plt.figure()
    ax = fig.add_subplot(3, 3, 7)
    ax.plot(t, total_accuracy_result[:, 0], color='k', label='RF')
    ax.plot(t, total_accuracy_result[:, 1], color='b', label='NB')
    ax.plot(t, total_accuracy_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_accuracy_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_accuracy_result[:, 4], color='c', label='GB')
    #ax.plot(t, total_accuracy_result[:, 5], color='m', label='SVM')
    ax.set_title("4-gram accuracy")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$accuracy$")
    ax.legend(loc='upper left')

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(t, total_precision_result[:, 0], color='k', label='RF')
    ax.plot(t, total_precision_result[:, 1], color='b', label='NB')
    ax.plot(t, total_precision_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_precision_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_precision_result[:, 4], color='c', label='GB')
    #ax.plot(t, total_precision_result[:, 5], color='m', label='SVM')
    ax.set_title("4-gram precision")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$precision$")
    ax.legend(loc='upper left')

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(t, total_f1_result[:, 0], color='k', label='RF')
    ax.plot(t, total_f1_result[:, 1], color='b', label='NB')
    ax.plot(t, total_f1_result[:, 2], color='g', label='KNN')
    ax.plot(t, total_f1_result[:, 3], color='r', label='DTC')
    ax.plot(t, total_f1_result[:, 4], color='c', label='GB')
    #ax.plot(t, total_f1_result[:, 5], color='m', label='SVM')
    ax.set_title("4-gram f1-score")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$f1$")
    ax.legend(loc='upper left')

def _best_2gram(t):
    _2gram_selected_df = pd.read_pickle("./2gram_selected.pkl")
    _2gram_selected_test_df = pd.read_pickle("./2gram_selected_test.pkl")
    accuracy=[]
    precision=[]
    f1=[]

    for feature in tqdm(t, mininterval=1):
        print("\n\n#######################{0}#######################".format(feature))
        X_train = _2gram_selected_df.iloc[:-1, :feature]
        X_test = _2gram_selected_test_df.iloc[:-1, :feature]
        y_train = _2gram_selected_df.iloc[:-1, -1]
        y_test = _2gram_selected_test_df.iloc[:-1, -1]
        params = {
            'n_estimators': [50, 100, 150],
            'max_depth': [1, 3, 5],
            'min_samples_leaf': [1, 3],
            'min_samples_split': [2, 4]
        }

        gb = GradientBoostingClassifier(random_state=0)
        grid_cv=GridSearchCV(gb, param_grid=params, cv=2, n_jobs=-1)
        best_model=grid_cv.fit(X_train, y_train)
        print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
        print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))

        #best_model=GradientBoostingClassifier(random_state=0)
        best_model.fit(X_train, y_train)
        pred = best_model.predict(X_test)
        accuracy.append(accuracy_score(y_test, pred))
        precision.append(precision_score(y_test, pred, average='weighted', zero_division=0))
        f1.append(f1_score(y_test, pred, average='weighted'))
        print("Gradient Boosting Classifier accuracy:{0:.4f}".format(accuracy_score(y_test,pred)))
        print("Gradient Boosting Classifier precision:{0:.4f}".format(precision_score(y_test, pred, average='weighted', zero_division=0)))
        print("Gradient Boosting Classifier f1-score:{0:.4f}".format(f1_score(y_test, pred, average='weighted')))
    accuracy=np.array(accuracy)
    precision=np.array(precision)
    f1=np.array(f1)

    # fig=plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(t, accuracy)
    ax.set_title("2-gram accuracy")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$accuracy$")

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(t, precision)
    ax.set_title("2-gram precision")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$precision$")

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t, f1)
    ax.set_title("2-gram f1-score")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$f1-score$")


def _best_4gram(t):
    _4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
    _4gram_selected_test_df = pd.read_pickle("./4gram_selected_test.pkl")

    X_train = _4gram_selected_df.iloc[:-1, :t]
    X_test = _4gram_selected_test_df.iloc[:-1, :t]
    y_train = _4gram_selected_df.iloc[:-1, -1]
    y_test = _4gram_selected_test_df.iloc[:-1, -1]
    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [1, 3, 5],
        'min_samples_leaf': [1, 3],
        'min_samples_split': [2, 4]
    }

    gb = GradientBoostingClassifier(random_state=0)
    grid_cv=GridSearchCV(gb, param_grid=params, cv=2, n_jobs=-1, verbose=3)
    best_model=grid_cv.fit(X_train, y_train)
    print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
    print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))

    #best_model=GradientBoostingClassifier(random_state=0)
    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)

    print("Gradient Boosting Classifier accuracy:{0:.4f}".format(accuracy_score(y_test,pred)))
    print("Gradient Boosting Classifier precision:{0:.4f}".format(precision_score(y_test, pred, average='weighted', zero_division=0)))
    print("Gradient Boosting Classifier f1-score:{0:.4f}".format(f1_score(y_test, pred, average='weighted')))

def _best_t_for_2gram(t):
    print("#################2gram_result#################\n")

    _2gram_selected_df = pd.read_pickle("./2gram_selected.pkl")
    _2gram_selected_test_df = pd.read_pickle("./2gram_selected_test.pkl")
    total_accuracy_result = []
    for feature in tqdm(t, mininterval=1):
        X_train = _2gram_selected_df.iloc[:-1, :feature]
        X_test = _2gram_selected_test_df.iloc[:-1, :feature]
        y_train = _2gram_selected_df.iloc[:-1, -1]
        y_test = _2gram_selected_test_df.iloc[:-1, -1]

        gb = GradientBoostingClassifier(random_state=0)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_test)

        total_accuracy_result.append(accuracy_score(y_test, pred))


    total_accuracy_result = np.array(total_accuracy_result)

    print("##accuracy#####################################\n")
    print(total_accuracy_result)

    fig=plt.figure()
    plt.plot(t, total_accuracy_result, color='c', label='GB')
    plt.title("2-gram accuracy")
    plt.xlabel("$t$")
    plt.ylabel("$accuracy$")
    plt.legend(loc='upper left')
    plt.show()


def _best_t_for_4gram(t):
    print("#################4gram_result#################\n")

    _4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
    _4gram_selected_test_df = pd.read_pickle("./4gram_selected_test.pkl")
    total_accuracy_result = []
    for feature in tqdm(t, mininterval=1):
        X_train = _4gram_selected_df.iloc[:-1, :feature]
        X_test = _4gram_selected_test_df.iloc[:-1, :feature]
        y_train = _4gram_selected_df.iloc[:-1, -1]
        y_test = _4gram_selected_test_df.iloc[:-1, -1]

        gb = GradientBoostingClassifier(random_state=0)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_test)

        total_accuracy_result.append(accuracy_score(y_test, pred))


    total_accuracy_result = np.array(total_accuracy_result)

    print("##accuracy#####################################\n")
    print(total_accuracy_result)

    fig=plt.figure()
    plt.plot(t, total_accuracy_result, color='c', label='GB')
    plt.title("4-gram accuracy")
    plt.xlabel("$t$")
    plt.ylabel("$accuracy$")
    plt.legend(loc='upper left')
    plt.show()





def _2gram(t):
    _2gram_selected_df = pd.read_pickle("./2gram_selected.pkl")
    _2gram_selected_test_df = pd.read_pickle("./2gram_selected_test.pkl")

    X_train = _2gram_selected_df.iloc[:-1, :t]
    X_test = _2gram_selected_test_df.iloc[:-1, :t]
    y_train = _2gram_selected_df.iloc[:-1, -1]
    y_test = _2gram_selected_test_df.iloc[:-1, -1]

    gb = GradientBoostingClassifier(random_state=0,n_estimators=150, max_depth=5, min_samples_leaf=1, min_samples_split=4, verbose=2)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    prob=gb.predict_proba(X_test)[:,1]
    false_positive_rate, true_positive_rate, threshold=roc_curve(y_test, prob)

    plt.title("Receiver Operating Characteristic")
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0,1], ls="--")
    plt.plot([0,0], [1,0], c=".7"), plt.plot([1,1], c=".7")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()

    precision, recall, threshold=precision_recall_curve(y_test, prob)

    plt.title("Precision-Recall Curve")
    plt.plot(precision, recall)
    plt.plot([0,1], ls="--")
    plt.plot([1,1], c=".7"), plt.plot([1,1],[1,0], c=".7")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.show()


    print("Gradient Boosting Classifier accuracy:{0:.4f}".format(accuracy_score(y_test, pred)))
    print("Gradient Boosting Classifier precision:{0:.4f}".format(
        precision_score(y_test, pred, average='weighted', zero_division=0)))
    print("Gradient Boosting Classifier f1-score:{0:.4f}".format(f1_score(y_test, pred, average='weighted')))

    print("Gradient Boosting Classifier ROCAUC:{0:.4f}".format(roc_auc_score(y_test, prob)))
    print("Gradient Boosting Classifier PRAUC:{0:.4f}".format(average_precision_score(y_test, prob)))

def _4gram(t):
    _4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
    _4gram_selected_test_df = pd.read_pickle("./4gram_selected_test.pkl")

    X_train = _4gram_selected_df.iloc[:-1, :t]
    X_test = _4gram_selected_test_df.iloc[:-1, :t]
    y_train = _4gram_selected_df.iloc[:-1, -1]
    y_test = _4gram_selected_test_df.iloc[:-1, -1]

    gb = GradientBoostingClassifier(random_state=0,n_estimators=150, max_depth=3, min_samples_leaf=3, min_samples_split=2, verbose=2)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    prob=gb.predict_proba(X_test)[:,1]
    false_positive_rate, true_positive_rate, threshold=roc_curve(y_test, prob)

    plt.title("Receiver Operating Characteristic")
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0,1], ls="--")
    plt.plot([0,0], [1,0], c=".7"), plt.plot([1,1], c=".7")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()

    precision, recall, threshold=precision_recall_curve(y_test, prob)

    plt.title("Precision-Recall Curve")
    plt.plot(precision, recall)
    plt.plot([0,1], ls="--")
    plt.plot([1,1], c=".7"), plt.plot([1,1],[1,0], c=".7")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.show()


    print("Gradient Boosting Classifier accuracy:{0:.4f}".format(accuracy_score(y_test, pred)))
    print("Gradient Boosting Classifier precision:{0:.4f}".format(
        precision_score(y_test, pred, average='weighted', zero_division=0)))
    print("Gradient Boosting Classifier f1-score:{0:.4f}".format(f1_score(y_test, pred, average='weighted')))

    print("Gradient Boosting Classifier ROCAUC:{0:.4f}".format(roc_auc_score(y_test, prob)))
    print("Gradient Boosting Classifier PRAUC:{0:.4f}".format(average_precision_score(y_test, prob)))

def main():
    #_2gram_cross_validation(range(100, 1000, 100))
    #_3gram_cross_validation(range(100, 1000, 100))
    #_4gram_cross_validation(range(100, 1000, 100))

    #_2gram_t(range(100, 7000, 500))
    #_3gram_t(range(100, 7000, 500))
    #_4gram_t(range(100, 7000, 500))

    #_best_t_for_2gram(range(100,7000,200))
    #_best_2gram([2100,3700,3900,4300,4500,5100,5500,5900,6100])
    _2gram(5900)

if __name__=='__main__':
    fig = plt.figure()
    main()
    plt.show()
