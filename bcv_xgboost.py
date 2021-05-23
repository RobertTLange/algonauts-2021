import pandas as pd
import numpy as np
import xgboost as xgb
# import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold, RepeatedKFold

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# SETTINGS - CHANGE THESE TO GET SOMETHING MEANINGFUL
ITERATIONS = 5 # 1000
CV_FOLDS = 3
XGB_THREADS = 1
BCV_THREADS = 3

model = xgb.XGBRegressor(
    n_jobs = XGB_THREADS,
    objective = 'reg:squarederror',
    eval_metric = 'logloss',
    tree_method='approx',
    verbosity = 0
)
#model = MultiOutputRegressor(model)
# 'estimator__objective', 'estimator__base_score', 'estimator__booster',
# 'estimator__colsample_bylevel', 'estimator__colsample_bynode',
# 'estimator__colsample_bytree', 'estimator__gamma', 'estimator__gpu_id',
# 'estimator__importance_type', 'estimator__interaction_constraints',
# 'estimator__learning_rate', 'estimator__max_delta_step', 'estimator__max_depth',
# 'estimator__min_child_weight', 'estimator__missing', 'estimator__monotone_constraints',
# 'estimator__n_estimators', 'estimator__n_jobs', 'estimator__num_parallel_tree',
# 'estimator__random_state', 'estimator__reg_alpha', 'estimator__reg_lambda',
# 'estimator__scale_pos_weight', 'estimator__subsample', 'estimator__tree_method',
# 'estimator__validate_parameters', 'estimator__verbosity', 'estimator__eval_metric',
# 'estimator__silent', 'estimator', 'n_jobs'

bayes_cv_tuner = BayesSearchCV(
    estimator = model,
    search_spaces = {
        'estimator__learning_rate': (0.01, 1.0, 'log-uniform'),
        'estimator__min_child_weight': (0, 10),
        'estimator__max_depth': (0, 50),
        'estimator__max_delta_step': (0, 20),
        'estimator__subsample': (0.01, 1.0, 'uniform'),
        'estimator__colsample_bytree': (0.01, 1.0, 'uniform'),
        'estimator__colsample_bylevel': (0.01, 1.0, 'uniform'),
        'estimator__reg_lambda': (1e-9, 1000., 'log-uniform'),
        'estimator__reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'estimator__gamma': (1e-9, 0.5, 'log-uniform'),
        'estimator__min_child_weight': (0, 5),
        'estimator__n_estimators': (50, 100),
        'estimator__scale_pos_weight': (1e-6, 500., 'log-uniform')
    },
    scoring = 'neg_mean_squared_error',
    cv = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=1, random_state=42),
    n_jobs = BCV_THREADS,
    n_iter = ITERATIONS,
    verbose = 0,
    refit = True,
    random_state = 42
)


def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest Score: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=5000, n_features=10, n_informative=8,
                           n_targets=1, random_state=1, noise=0.1)
    result = bayes_cv_tuner.fit(X, y, callback=status_print)
    print(result)
    print(bayes_cv_tuner.cv_result)
