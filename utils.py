# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: utils.py
@Time: 2018/11/6 10:48
@Software: PyCharm 
@Description:
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def get_models():
    """
    生成机器学习库
    :return:
    """
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    ln_svc = LinearSVC()
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    nn = MLPClassifier()
    ab = AdaBoostClassifier()
    gb = GradientBoostingClassifier()
    rf = RandomForestClassifier()
    xgb = XGBClassifier()
    lgb = LGBMClassifier()
    models = {
        'naive bayes': nb,
        # 'svm': svc,
        # 'linear_svm': ln_svc,
        'knn': knn,
        'logistic': lr,
        # 'mlp-nn': nn,
        'ada boost':ab,
        'random forest': rf,
        'gradient boost': gb,
        'xgb':xgb,
        'lgb':lgb
    }
    return models


def score_models(models, X,y):
    """Score model in prediction DF"""
    print("评价每个模型.")
    for name,model in models.items():
        score = cross_val_score(model,X,y,scoring='roc_auc',cv=5)
        mean_score=np.mean(score)
        print("{}: {}" .format(name, mean_score))
    print("Done.\n")


