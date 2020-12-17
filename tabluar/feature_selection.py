## Null importance
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import time
from lightgbm import LGBMClassifier
import lightgbm as lgb

def get_feature_importances(data,features, target, shuffle, seed=None):
    # Shuffle target if required
    y = data[target].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data[target].copy().sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[features], y, free_raw_data=False, silent=False)
    lr = 0.01
    Early_Stopping_Rounds = 150

    N_round = 1500
    Verbose_eval = 100

    params =  {
        'n_estimators':N_round,
        'num_leaves': 61,  # 当前base 61
        'min_child_weight': 0.03454472573214212,
        'feature_fraction': 0.3797454081646243,
        'bagging_fraction': 0.4181193142567742,
        'min_data_in_leaf': 96,  # 当前base 106
        'objective': 'regression',
        "metric": 'rmse',
        'max_depth': -1,
        'learning_rate': lr,   # 快速验证
#              'learning_rate': 0.006883242363721497,
        "boosting_type": "gbdt",
        "bagging_seed": 11,
        "verbosity": -1,
        'reg_alpha': 0.3899927210061127,
        'reg_lambda': 0.6485237330340494,
        'random_state': 47,
        'num_threads': 16,
        'lambda_l1': 1,  
        'lambda_l2': 1,
    #     'early_stopping_rounds':Early_Stopping_Rounds,
        'verbose': Verbose_eval,
    #     'is_unbalance':True
    }
    
    
    # Fit the model
    clf = lgb.train(params=params, train_set=dtrain)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    # imp_df['trn_score'] = sqrt(mean_squared_error(y, clf.predict(data[features])))
    
    return imp_df
def get_actual_imp_df(train_df, features):
    # Seed the unexpected randomness of this world
    np.random.seed(817)
    # Get the actual importance, i.e. without shuffling
    actual_imp_df = get_feature_importances(data=train_df,features=features, target='accuracy_group_target', shuffle=False)
    return actual_imp_df

def null_imp_df(train_df, features, nb_runs):

    null_imp_df = pd.DataFrame()
    nb_runs = 20
    import time
    start = time.time()
    dsp = ''
    for i in range(nb_runs):
        # Get current run importances
        imp_df = get_feature_importances(data=train_df,features=features, target='accuracy_group_target', shuffle=True)
        imp_df['run'] = i + 1 
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        # Erase previous message
        for l in range(len(dsp)):
            print('\b', end='', flush=True)
        # Display current run and time used
        spent = (time.time() - start) / 60
        dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
        print(dsp, end='', flush=True)
    return null_imp_df

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
%matplotlib inline

feature_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    # act_importance should be much bigger than null importance
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
    feature_scores.append((_f, split_score, gain_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(1, 2)
# Plot Split importances
# ax = plt.subplot(gs[0, 0])
# sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:100], ax=ax)
# ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
# Plot Gain importances
ax = plt.subplot(gs[0, 1])
sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:100], ax=ax)
ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
plt.tight_layout()

pd.set_option('max_rows',2000)
new_list = scores_df.sort_values(by=['gain_score'],ascending=False).reset_index(drop=True)
new_list.head(2000)

for item in new_list['feature']:
    #print (item) 
    print ('"' + str(item) +  '",') 
    
null_importance_feature = []  
