from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from encoding_models.trees import fit_gradboost_model, gb_params_to_search
from encoding_models.trees import fit_linear_model, lm_params_to_search


def get_hyperspace(params_to_search):
    """ Helper fct to construct skopt hyperparam space. """
    param_range = {}
    if "categorical" in params_to_search.keys():
        for k, v in params_to_search["categorical"].items():
            param_range[k] = Categorical(v, name=k)
    if "real" in params_to_search.keys():
        for k, v in params_to_search["real"].items():
            param_range[k] = Real(float(v["begin"]), float(v["end"]),
                                  prior=v["prior"], name=k)
    if "integer" in params_to_search.keys():
        for k, v in params_to_search["integer"].items():
            param_range[k] = Integer(int(v["begin"]), int(v["end"]),
                                     prior=v["prior"], name=k)
    return param_range


def run_bayes_opt(smbo_config, params_to_search, X, y):
    """ Simple loop running SMBO + Cross-Validation. """
    param_range = get_hyperspace(params_to_search)
    hyper_optimizer = Optimizer(dimensions=list(param_range.values()),
                                random_state=1,
                                base_estimator=smbo_config["base_estimator"],
                                acq_func=smbo_config["acq_function"],
                                n_initial_points=smbo_config["n_initial_points"])
    for t in range(smbo_config["opt_iters"]):
        proposal = hyper_optimizer.ask(n_points=1)[0]
        model_config = {}
        for i, k in enumerate(param_range.keys()):
            model_config[k] = proposal[i]
        scores = fit_linear_model(model_config, X, y)
        hyper_optimizer.tell(proposal, scores)
        #print(t, model_config, scores, proposal)
        print(50*"=")
        print(t, hyper_optimizer.get_result().fun, scores)
        print(hyper_optimizer.get_result().x)
    return hyper_optimizer


def get_data():
    from sklearn import preprocessing
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=2000, n_informative=5,
                           n_targets=100, random_state=1, noise=1)
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y


def main():
    X, y = get_data()
    from encoding_models.ols import OLS_pytorch
    import numpy as np
    ols = OLS_pytorch(1e-5)
    ols.fit(X, y.T)
    preds = ols.predict(X)
    score = np.mean((y - preds)**2)
    print(score)
    smbo_config = {"base_estimator": "GP",      # "GP", "RF", "ET", "GBRT"
                   "acq_function": "gp_hedge",  # "LCB", "EI", "PI", "gp_hedge"
                   "n_initial_points": 5,
                   "opt_iters": 1}
    result = run_bayes_opt(smbo_config, lm_params_to_search, X, y)


if __name__ == "__main__":
    main()
