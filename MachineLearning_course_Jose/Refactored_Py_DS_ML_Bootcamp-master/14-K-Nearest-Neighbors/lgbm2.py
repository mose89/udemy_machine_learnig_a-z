# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:32:21 2019

@author: georg
"""


from imblearn.under_sampling import TomekLinks
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os
import gc
gc.enable()
#from imblearn.over_sampling import SMOTE
#from sklearn.model_selection import KFold
from imblearn.under_sampling import CondensedNearestNeighbour


    
    
def fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name):



    
    model = lgb.LGBMClassifier(
               n_estimators=999999,
               objective = "binary", 
               boost="gbdt",
               metric="auc",
               boost_from_average="false",
               num_threads=28,
               learning_rate = 0.01,
               num_leaves = 13,
               max_depth=-1,
               tree_learner = "serial",
               feature_fraction = 0.05,
               bagging_freq = 5,
               bagging_fraction = 0.4,
               min_data_in_leaf = 80,
               min_sum_hessian_in_leaf = 10.0,
               verbosity = 1)
    
    
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)], 
              verbose=0, early_stopping_rounds=4000)
              
    cv_val = model.predict_proba(X_val)[:,1]
    
    save_to = '{}{}_fold{}.txt'.format(cb_path, name, counter+1)
    model.booster_.save_model(save_to)    
    
                     
    return cv_val


def train_stage(df_path, cb_path):
    
    print('Load Train Data.')
    df = pd.read_csv(df_path)
    print('\nShape of Train Data: {}'.format(df.shape))
    
    y_df = np.array(df['target'])                        
    df_ids = np.array(df.index)                     
    df.drop(['ID_code', 'target'], axis=1, inplace=True)
    
    cb_cv_result  = np.zeros(df.shape[0])
    
    skf = StratifiedKFold(n_splits=15, shuffle=False, random_state=42)
    skf.get_n_splits(df_ids, y_df)
    



    #sm = TomekLinks(random_state=42)
    sm = CondensedNearestNeighbour(random_state=42, n_jobs = 3)

    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df.values[ids[1]], y_df[ids[1]]

        X_fit, y_fit = sm.fit_sample(X_fit, y_fit)
    
        print('CatBoost')
        cb_cv_result[ids[1]]  += fit_cb(X_fit,  y_fit, X_val, y_val, counter, cb_path,  name='cb')
        
        del X_fit, X_val, y_fit, y_val
        gc.collect()
    
    auc_cb   = round(roc_auc_score(y_df, cb_cv_result), 4)
    print('Catboost VAL AUC: {}'.format(auc_cb))
    
    return 0
    
    
def prediction_stage(df_path, cb_path):
    
    print('Load Test Data.')
    df = pd.read_csv(df_path)
    print('\nShape of Test Data: {}'.format(df.shape))
    
    df.drop(['ID_code'], axis=1, inplace=True)
    
    cb_models  = sorted(os.listdir(cb_path))
    
    cb_result  = np.zeros(df.shape[0])
    
    print('\nMake predictions...\n')
    print('With CatBoost...')        
    for m_name in cb_models:
        
        
        model = lgb.Booster(model_file='{}{}'.format(cb_path, m_name))
        cb_result += model.predict(df.values)        
        
        
    
    cb_result  /= len(cb_models)
    
    submission = pd.read_csv(r"C:\Users\georg\Documents\sample_submission.csv")
    submission['target'] = cb_result
    submission.to_csv('cb_starter_submission.csv', index=False)
    return 0
    
    
if __name__ == '__main__':
    
    train_path = r"C:\Users\georg\Documents\kaggle_train.csv"
    test_path  = r"C:\Users\georg\Documents\kaggle_test.csv"
    
    
    cb_path  = './cb_models_stack59/'

    os.mkdir(cb_path)

    print('Train Stage.\n')
    train_stage(train_path, cb_path)
    
    print('Prediction Stage.\n')
    prediction_stage(test_path, cb_path)
    
    print('\nDone.')
    
    

    
    
    
    