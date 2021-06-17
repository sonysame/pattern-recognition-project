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
                                        min_samples_split=8, random_state=0)
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
        knn = KNeighborsClassifier(n_neighbors=8)
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
                                        min_samples_split=8, random_state=0)
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
        knn = KNeighborsClassifier(n_neighbors=8)
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
                                        min_samples_split=8, random_state=0)
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
        knn = KNeighborsClassifier(n_neighbors=8)
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


def _best_4gram():
    _4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
    _4gram_selected_test_df = pd.read_pickle("./4gram_selected_test.pkl")

    accuracy_result = []
    X_train = _4gram_selected_df.iloc[:-1, :34500]
    X_test = _4gram_selected_test_df.iloc[:-1, :34500]
    y_train = _4gram_selected_df.iloc[:-1, -1]
    y_test = _4gram_selected_test_df.iloc[:-1, -1]

    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)

    print("Gradient Boosting Classifier accuracy:{0:.4f}".format(accuracy_score(y_test,pred)))
    print("Gradient Boosting Classifier precision:{0:.4f}".format(precision_score(y_test, pred, average='weighted', zero_division=0)))
    print("Gradient Boosting Classifier f1-score:{0:.4f}".format(f1_score(y_test, pred, average='weighted')))


def _4gram():
    _4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
    _4gram_selected_test_df = pd.read_pickle("./4gram_selected_test.pkl")

    accuracy_result = []
    X_train = _4gram_selected_df.iloc[:-1, :32500]
    X_test = _4gram_selected_test_df.iloc[:-1, :32500]
    y_train = _4gram_selected_df.iloc[:-1, -1]
    y_test = _4gram_selected_test_df.iloc[:-1, -1]

    params = {
        'n_estimators': [300, 400, 500],
        'max_depth': [6, 8, 10, 12],
        'min_samples_leaf': [8, 12, 16],
        'min_samples_split': [8, 12, 16]
    }
    """
    rf_clf=RandomForestClassifier(random_state=0, n_jobs=-1)
    grid_cv=GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
    grid_cv.fit(X_train, y_train)

    print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
    print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))
    """
    rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=500, max_depth=6, min_samples_leaf=8,
                                    min_samples_split=8, random_state=0)
    rf_clf.fit(X_train, y_train)
    pred = rf_clf.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))

    print("Random Forest 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))

    gnb = GaussianNB()
    pred = gnb.fit(X_train, y_train).predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("GaussianNB 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))

    """
    knn_result=[]
    for k in range(1,20):
        knn= KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        pred=knn.predict(X_test)
        knn_result.append(accuracy_score(y_test,pred))
        print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
    fig=plt.figure()
    plt.plot(range(1,20), knn_result)
    plt.title("Best k for KNN")
    plt.xlabel("$k$")
    plt.ylabel("$accuracy$")
    fig.show()

    """
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))

    dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
    dtc.fit(X_train, y_train)
    pred = dtc.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("Decision Tree Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))

    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))

    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("SVM 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))


def main():

    _2gram_t(range(100, 7012, 1000))
    _3gram_t(range(100, 61225, 10000))
    _4gram_t(range(100, 248883, 30000))

    #_3gram_t(range(2000,3000,100))
    #_4gram_t(range(30000,35000,500))
    #_best_4gram()
    #_4gram()
if __name__=='__main__':
    fig = plt.figure()
    main()
    plt.show()
