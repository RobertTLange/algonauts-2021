from encoding_models.trees import fit_gradboost_model, gb_params_to_search
from encoding_models.ols import fit_linear_model, lm_params_to_search
from encoding_models.networks import fit_mlp_model, mlp_params_to_search

from skopt import Optimizer
from skopt.space import Real, Integer, Categorical


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


def run_bayes_opt(mle, param_hyperspace, X, y):
    """ Simple loop running SMBO + Cross-Validation. """
    hyper_optimizer = Optimizer(dimensions=list(param_hyperspace.values()),
                                random_state=1,
                                **mle.train_config.smbo_config)
    for t in range(mle.train_config["bo_opt_iters"]):
        proposal = hyper_optimizer.ask()
        model_config = {}
        for i, k in enumerate(param_hyperspace.keys()):
            model_config[k] = proposal[i]

        if mle.net_config.encoding_model == "linear_regression":
            cv_score_mean, cv_score_std  = fit_linear_model(model_config, X, y,
                                                            mle.train_config.cv_folds)

        hyper_optimizer.tell(proposal, cv_score_mean)

        time_tick = {"total_bo_iters": t+1}
        stats_tick = {"cv_score_mean": cv_score_mean,
                      "cv_score_std": cv_score_std,
                      "best_bo_score": hyper_optimizer.get_result().fun}
        for i, k in enumerate(param_hyperspace):
            stats_tick[k] = hyper_optimizer.get_result().x[i]
        mle.update_log(time_tick, stats_tick, save=True)
    return hyper_optimizer
