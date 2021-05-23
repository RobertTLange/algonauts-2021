import xgboost as xgb
from tpot import TPOTRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# TODO: Add lightbm model! Faster

def fit_xgb_model(model_config, X, y):
    model = xgb.XGBRegressor(
        n_jobs = 4,
        objective = 'reg:squarederror',
        eval_metric = 'logloss',
        tree_method='auto',
        verbosity = 0,
        **model_config
    )
    model = MultiOutputRegressor(model)
    model = model.fit(X, y)
    preds = model.predict(X)
    return np.mean((y - preds)**2)


def fit_gradboost_model(model_config, X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
    model = GradientBoostingRegressor(loss='ls', min_samples_leaf=9,
                                      min_samples_split=9, **model_config)
    wrapper = MultiOutputRegressor(model)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(wrapper, X, y,
                               scoring='neg_mean_absolute_error',
                               cv=cv, n_jobs=5)
    # force the scores to be positive
    n_scores = np.absolute(n_scores)
    # summarize performance
    print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    # model = model.fit(X, y)
    # preds = model.predict(X)
    # return np.mean((y - preds)**2)
    return np.mean(n_scores)


def fit_teapot_model(model_config, X, y):
    model = TPOTRegressor(generations=model_config["num_generations"],
                          population_size=model_config["population_size"],
                          n_jobs=5)
    model = MultiOutputRegressor(model)
    model = model.fit(X, y)
    preds = model.predict(X)
    return np.mean((y - preds)**2)


# XGBoost Hyperparameters
xgb_params_to_search = {
"real":
    {"learning_rate": {"begin": 0.01, "end": 1.0, "prior": 'log-uniform'},
     "subsample": {"begin": 0.01, "end": 1.0, "prior": 'log-uniform'},
     "colsample_bytree": {"begin": 0.01, "end": 1.0, "prior": 'uniform'},
     'colsample_bytree': {"begin": 0.01, "end": 1.0, "prior": 'uniform'},
     'colsample_bylevel': {"begin": 0.01, "end": 1.0, "prior": 'uniform'},
     'reg_lambda': {"begin": 1e-9, "end": 1000., "prior": 'log-uniform'},
     'reg_alpha': {"begin": 1e-9, "end": 1.0, "prior": 'log-uniform'},
     'gamma': {"begin": 1e-9, "end": 0.5, "prior": 'log-uniform'},
     'scale_pos_weight': {"begin": 1e-6, "end": 500., "prior": 'log-uniform'}
     },
"integer":
    {'min_child_weight': {"begin": 0, "end": 10, "prior": 'uniform'},
     'max_depth': {"begin": 1, "end": 50, "prior": 'uniform'},
     'max_delta_step': {"begin": 0, "end": 20, "prior": 'uniform'},
     'min_child_weight': {"begin": 0, "end": 5, "prior": 'uniform'},
     'n_estimators': {"begin": 50, "end": 500, "prior": 'uniform'},
     }
}

# Sklearn GradientBoostingRegressor Hyperparameters
gb_params_to_search = {
"real":
    {"alpha": {"begin": 0.01, "end": 0.999, "prior": 'log-uniform'},
     "learning_rate": {"begin": 0.01, "end": 0.99, "prior": 'log-uniform'}
     },
"integer":
    {'max_depth': {"begin": 1, "end": 50, "prior": 'uniform'},
     'n_estimators': {"begin": 50, "end": 500, "prior": 'uniform'},
     }
}

# Linear Regression Hyperparameters
lm_params_to_search = {
"real":
    {"lambda_reg": {"begin": 1e-07, "end": 1e-1, "prior": 'log-uniform'}},
}
