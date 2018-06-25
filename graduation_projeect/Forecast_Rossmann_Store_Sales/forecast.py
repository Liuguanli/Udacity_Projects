#coding=utf-8

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt

from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def build_features(features, data):
    
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)
    
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    
    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear
    
    # CompetitionOpenSinceYear 和 CompetitionOpenSinceMonth为空使用平均值
    features.append('CompetitionOpen')
    
    data.loc[data.CompetitionOpenSinceYear == 0, 'CompetitionOpenSinceYear'] = data.loc[data.CompetitionOpenSinceYear != 0, 'CompetitionOpenSinceYear'].mean()
    data.loc[data.CompetitionOpenSinceMonth == 0, 'CompetitionOpenSinceMonth'] = data.loc[data.CompetitionOpenSinceMonth != 0, 'CompetitionOpenSinceMonth'].mean()
#     data.loc[data.CompetitionOpenSinceYear == 0, 'CompetitionOpenSinceYear'] = data.loc[data.CompetitionOpenSinceYear != 0, 'CompetitionOpenSinceYear'].median()
#     data.loc[data.CompetitionOpenSinceMonth == 0, 'CompetitionOpenSinceMonth'] = data.loc[data.CompetitionOpenSinceMonth != 0, 'CompetitionOpenSinceMonth'].median()
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + (data.Month - data.CompetitionOpenSinceMonth)
    
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) +  (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
#     data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0
    
    features.append('IsPromoMonth')
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

	features.append('SalesPerDay')
    for store in data.Store.unique():
    	data.loc[data.Store == store, 'SalesPerDay'] = data.loc[data.Store == store, 'Sales'].mean()
    
    return data

def train_model(alg, features, train, useTrainCV=True, cv_folds=5, early_stopping_rounds=100):
    num_boost_round = 1

    print("Train a XGBoost model")
    X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
    y_train = np.log1p(X_train.Sales)
    y_valid = np.log1p(X_valid.Sales)
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)
    
#     dxtrain = xgb.DMatrix(train[features], np.log1p(train.Sales))

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=num_boost_round, nfold=cv_folds,metrics='rmse', 
        verbose_eval=True, early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
#     alg.fit(X_train[features], y_train, eval_metric=rmspe_xg)
    print("fit begin")
    alg.fit(X_train[features], y_train, eval_metric='rmse')
    print("fit finish")
    
#     #Predict training set:
#     dtrain_predictions = alg.predict(X_valid[features])
#     dtrain_predprob = alg.predict_proba(X_valid[features])[:,1]

#     #Print model report:
#     print "\nModel Report"
#     print(dtrain_predictions)
#     print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    
#     watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
#     gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)
    return alg

# importing train data to learn
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}
train = pd.read_csv("./dataset/train.csv", parse_dates=[2], dtype=types)
test = pd.read_csv("./dataset/test.csv", parse_dates=[3], dtype=types)
store = pd.read_csv("./dataset/store.csv")

train.fillna(1, inplace=True)
test.fillna(1, inplace=True)

train = train[(train["Open"] != 0) & (train['Sales'] != 0)]

train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')
features = []
build_features(features, train)
build_features([], test)


param_test1 = {
 'max_depth':range(5,10,1),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(         learning_rate =0.1, n_estimators=140, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4,     scale_pos_weight=1, seed=27), 
 param_grid = param_test1,  scoring=rmspe,   n_jobs=4,iid=False, cv=5)

gsearch1.fit(train[features],np.log1p(train.Sales))
gsearch1.grid_scores_, gsearch1.best_params_,     gsearch1.best_score_

# xgb1 = XGBClassifier(
#          learning_rate =0.1,
#          n_estimators=1000,
#          max_depth=5,
#          min_child_weight=1,
#          gamma=0,
#          subsample=0.8,
#          colsample_bytree=0.8,
#          objective= 'reg:linear',
#          nthread=4,
#          scale_pos_weight=1,
#          seed=27)

# gbm = train_model(xgb1, features, train)
# dtest = xgb.DMatrix(test[features])
# test_probs = gbm.predict(test[features])
# # test_probs = gbm.predict(dtest)
# result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
# result.to_csv("xgboost_submission.csv", index=False)



