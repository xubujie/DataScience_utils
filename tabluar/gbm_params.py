## https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

params = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    'num_iterations': 100, 
    "num_leaves": 16,
    "feature_fraction": 1.0,
    'bagging_fraction': 0.7, 
    "feature_fraction": 0.7,
    "verbosity": -1
}