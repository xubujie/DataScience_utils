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


def lgb_kfold(train_df,test_df,features,target,cat_features,folds,params):
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    cv_list = []
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[features], train_df['salary_round'])):
        print ('FOLD:' + str(n_fold))
        
        train_x, train_y = train_df[features].iloc[train_idx], train_df[target].iloc[train_idx]
        valid_x, valid_y = train_df[features].iloc[valid_idx], train_df[target].iloc[valid_idx]
        
        print ('train_x shape:',train_x.shape)
        print ('valid_x shape:',valid_x.shape)
        
        dtrain = lgb.Dataset(train_x, label=train_y,categorical_feature=cat_features)
        dval = lgb.Dataset(valid_x, label=valid_y, reference=dtrain,categorical_feature=cat_features) 
        bst = lgb.train(params, dtrain, num_boost_round=50000,
            valid_sets=[dval,dtrain], verbose_eval=500,early_stopping_rounds=500, ) 
        new_list = sorted(zip(features, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)[:]
        for item in new_list:
            print (item) 
         
        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        oof_cv = mean_absolute_error(valid_y,  oof_preds[valid_idx])
        cv_list.append(oof_cv)
        print (cv_list)
        sub_preds += bst.predict(test_df[features], num_iteration=bst.best_iteration) / folds.n_splits
 
    cv = mean_absolute_error(train_df[target],  oof_preds)
    print('Full OOF MAE %.6f' % cv)  

    train_df['lgb_y'] = oof_preds
    test_df['lgb_y'] = sub_preds
    
    return train_df,test_df