from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np


def get_hyperspace(params_to_search):
    param_range = {}
    if "categorical" in params_to_search.keys():
        for k, v in params_to_search["categorical"].items():
            param_range[k] = Categorical(v, name=k)
    if "real" in params_to_search.keys():
        for k, v in params_to_search["real"].items():
            param_range[k] = Real(float(v["begin"]), float(v["end"]),
                                  prior=v["prior"], name=k)
    if "integer" in params_to_search.keys():
            param_range[k] = Integer(int(v["begin"]), int(v["end"]),
                                     prior=v["prior"], name=k)
    return param_range


def fit_model(model_config, X, y):
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


def run_bayes_opt(smbo_config, params_to_search, X, y):
    param_range = get_hyperspace(params_to_search)
    hyper_optimizer = Optimizer(dimensions=list(param_range.values()),
                                random_state=1,
                                base_estimator=smbo_config["base_estimator"],
                                acq_func=smbo_config["acq_function"],
                                n_initial_points=smbo_config["n_initial_points"])
    for t in range(smbo_config["opt_iters"]):
        proposal = hyper_optimizer.ask(n_points=1)
        model_config = {}
        model_config["learning_rate"] = proposal[0][0]
        scores = fit_model(model_config, X, y)

        hyper_optimizer.tell(proposal[0], scores)
        print(t, model_config, scores, proposal)
        #print(hyper_optimizer.get_result())
        print(hyper_optimizer.get_result().fun, hyper_optimizer.get_result().x)
    return hyper_optimizer


if __name__ == "__main__":
    # def get_template_data():
    #     from sklearn import (datasets, preprocessing,
    #                          decomposition, metrics, multioutput)
    #     # For data we use 6 different digit classes of 8x8 pixels
    #     digits = datasets.load_digits(n_class=6)
    #     X = digits.data # (1083, 64)
    #     y = digits.target # (1083, )
    #     # First, PCA 2-D (which has .transform()) to illustrate and evaluate
    #     lens = decomposition.PCA(n_components=10, random_state=0)
    #     X_lens = lens.fit_transform(X)
    #     # Normalize the lens within 0-1
    #     scaler = preprocessing.MinMaxScaler()
    #     y = scaler.fit_transform(X_lens) + np.random.normal(size=X_lens.shape)
    #     return X, y
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=5000, n_features=200, n_informative=20,
                           n_targets=10, random_state=1, noise=1)

    from utils.ols import OLS_pytorch
    ols = OLS_pytorch()
    ols.fit(X, y.T)
    preds = ols.predict(X)
    print(np.mean((y - preds)**2))
    params_to_search = {"real":
    {"estimator__learning_rate": {"begin": 0.01, "end": 1.0, "prior": 'log-uniform'},
     "estimator__subsample": {"begin": 0.01, "end": 1.0, "prior": 'log-uniform'},
     "estimator__colsample_bytree": {"begin": 0.01, "end": 1.0, "prior": 'uniform'},}}

    # 'estimator__min_child_weight': (0, 10),
    # 'estimator__max_depth': (0, 50),
    # 'estimator__max_delta_step': (0, 20),
    # 'estimator__colsample_bytree': (0.01, 1.0, 'uniform'),
    # 'estimator__colsample_bylevel': (0.01, 1.0, 'uniform'),
    # 'estimator__reg_lambda': (1e-9, 1000., 'log-uniform'),
    # 'estimator__reg_alpha': (1e-9, 1.0, 'log-uniform'),
    # 'estimator__gamma': (1e-9, 0.5, 'log-uniform'),
    # 'estimator__min_child_weight': (0, 5),
    # 'estimator__n_estimators': (50, 100),
    # 'estimator__scale_pos_weight': (1e-6, 500., 'log-uniform')

    smbo_config = {"base_estimator": "GP",          # "GP", "RF", "ET", "GBRT"
                   "acq_function": "gp_hedge",      # "LCB", "EI", "PI", "gp_hedge"
                   "n_initial_points": 5,
                   "opt_iters": 50}

    # result = run_bayes_opt(smbo_config, params_to_search, X, y)
    from tpot import TPOTRegressor
    model = TPOTRegressor(generations=1, population_size=5, n_jobs=5)
    model = MultiOutputRegressor(model)
    model = model.fit(X, y)
    preds = model.predict(X)
    print(np.mean((y - preds)**2))
