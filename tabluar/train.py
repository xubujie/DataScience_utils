import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import os
import logging

def train_lgb(X, y, params, folder, rewrite=False, fold=5):
    os.makedirs(folder, exist_ok=rewrite)
    spl = StratifiedKFold(n_splits=fold, random_state=2019, shuffle=True)
    feature_importance_df = pd.DataFrame()
    counter = 0
    for trn_ids, val_ids in spl.split(X, y):
        counter += 1
        print('fold {}'.format(counter))
        X_trn, y_trn = X.iloc[trn_ids], y[trn_ids]
        X_val, y_val = X.iloc[val_ids], y[val_ids]
        lgb_trn = lgb.Dataset(X_trn, y_trn)
        lgb_val = lgb.Dataset(X_val, y_val)
        model = lgb.train(params=params, train_set=lgb_trn,verbose_eval=500,
                          valid_sets=[lgb_trn,lgb_val],valid_names=['train','valid'])
        model.save_model(folder + '/{}_{}'.format(counter, model.best_score['valid']))
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_trn.columns
        fold_importance_df["importance"] = model.feature_importance()
        fold_importance_df["fold"] = counter
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    feature_importance_df = feature_importance_df.groupby('feature', as_index=False)['importance'].mean().sort_values(by='importance', ascending=False)
    feature_importance_df.to_csv(os.path.join(folder, 'feature_importance.csv'), index=False)
    print('done')
