# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: main.py 
@Time: 2018/11/8 9:34
@Software: PyCharm 
@Description:
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

pd.set_option('display.max_columns',100)
df_train=pd.read_table('input/train.tsv')
y_encoder=LabelEncoder()
df_train['Y']=y_encoder.fit_transform(df_train['Y'])
df_test=pd.read_table('input/test.tsv')
train_len = len(df_train)
df = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)


def add_poly_features(data,column_names):
    # 组合特征
    features=data[column_names]
    rest_features=data.drop(column_names,axis=1)
    poly_transformer=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
    poly_features=pd.DataFrame(poly_transformer.fit_transform(features),columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1,col,poly_features[col])
    return rest_features


def process_label(df):
    # 数据预处理 类别编码
    cate_cols=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']

    # "occupation", "native-country", "workclass"
    for cate in ["occupation", "native-country", "workclass"]:
        df[cate].replace("?",df['occupation'].mode()[0])

    # df.replace(['Divorced', 'Married-AF-spouse',
    #               'Married-civ-spouse', 'Married-spouse-absent',
    #               'Never-married', 'Separated', 'Widowed'],
    #              ['not married', 'married', 'married', 'married',
    #               'not married', 'not married', 'not married'], inplace=True) # 降低

    # df['race'] = df['race'].apply(lambda el: 1 if el.strip() == "White" else 0) # 降低

    df['native-country'] = df['native-country'].apply(lambda el: 1 if el.strip() == "United-States" else 0)
    df = pd.get_dummies(df, columns=cate_cols)
    return df


def process_nums(df):
    # 数据预处理 数值型数据
    num_cols=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    df["log_age"] = np.log(df.age - df.age.min() + 1)
    df["log_fnlwgt"] = np.log(df.fnlwgt + 1)
    df["log_education-num"] = np.log(df['education-num'] + 1)
    df["log_capital-gain"] = np.log(df['capital-gain'] -df['capital-gain'] + 1)
    df["log_capital-loss"] = np.log(df['capital-loss'] -df['capital-loss'] + 1)
    df["log_hours-per-week"] = np.log(df['hours-per-week'] -df['hours-per-week'] + 1)
    scaler=StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols].values)

    # df=add_poly_features(df,num_cols)
    return df


def create_feature(df):
    # 数据预处理 类别编码
    new_df=process_label(df)
    # 数据预处理 数值型数据
    new_df=process_nums(new_df)
    new_train,new_test=new_df[:train_len],new_df[train_len:]
    print(new_train.shape,new_test.shape)
    print(list(new_train.columns))
    return new_train,new_test


# 调整参数
def tune_params(model,params,X,y):
    gsearch = GridSearchCV(estimator=model,param_grid=params, scoring='roc_auc',n_jobs=-1)
    gsearch.fit(X, y)
    # print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)
    print(gsearch.best_params_, gsearch.best_score_)
    return gsearch


# 特征重要性
def plot_fea_importance(classifier,X_train):
    plt.figure(figsize=(10,12))
    name = "xgb"
    indices = np.argsort(classifier.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],
                    x=classifier.feature_importances_[indices][:40],orient='h')
    g.set_xlabel("Relative importance", fontsize=12)
    g.set_ylabel("Features", fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title(name + " feature importance")
    plt.show()


def evaluate_cv5_lgb(train_df, test_df, cols, test=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_test = 0
    oof_train = np.zeros((train_df.shape[0],))
    for i, (train_index, val_index) in enumerate(kf.split(train_df[cols])):
        X_train, y_train = train_df.loc[train_index, cols], train_df['Y'].values[train_index]
        X_val, y_val = train_df.loc[val_index, cols], train_df['Y'].values[val_index]
        xgb = XGBClassifier(learning_rate=0.12,
                            max_depth=6,
                            min_child_weight=3,
                            ubsample=0.98,
                            colsample_bytree=0.6)
        xgb.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=100, eval_metric=['auc'], verbose=True)
        y_pred = xgb.predict(X_val)

        if test:
            y_test += xgb.predict(test_df.loc[:, cols])
        oof_train[val_index] = y_pred

        if i==0:
            plot_fea_importance(xgb,X_train)
    print(train_df.Y.values)
    accuracy = accuracy_score(train_df.Y.values, oof_train.round())
    y_test /= 5
    print('5 Fold accuracy:', accuracy)
    return y_test


if __name__ == '__main__':
    train,test=create_feature(df)
    cols = [col for col in train.columns if col not in ['id','Y']]
    y_test=evaluate_cv5_lgb(train,test,cols,True)
    test['Y']=y_test
    test['Y']=test['Y'].apply(lambda x:1 if x>0.5 else 0)
    test['Y']=y_encoder.inverse_transform(test['Y'].values)
    test[['id','Y']].to_csv('result/01_lgb_cv5.csv',columns=None, header=False, index=False)
