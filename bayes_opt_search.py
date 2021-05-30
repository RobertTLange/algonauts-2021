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


def run_bayes_opt(mle, fitter, param_hyperspace):
    """ Simple loop running SMBO + Cross-Validation. """
    hyper_optimizer = Optimizer(dimensions=list(param_hyperspace.values()),
                                random_state=1,
                                **mle.train_config.smbo_config)
    for t in range(mle.train_config["bo_opt_iters"]):
        proposal = hyper_optimizer.ask()
        model_config = {}
        for i, k in enumerate(param_hyperspace.keys()):
            model_config[k] = proposal[i]

        cv_score_mean, cv_score_std  = fitter.cv_fit(model_config)
        hyper_optimizer.tell(proposal,
                             cv_score_mean[mle.train_config.bo_eval_metric])

        time_tick = {"total_bo_iters": t+1}
        stats_tick = {"best_bo_score": hyper_optimizer.get_result().fun}
        for i, k in enumerate(param_hyperspace):
            stats_tick[k] = proposal[i]
        for i, k in enumerate(cv_score_mean):
            stats_tick[k] = cv_score_mean[k]
        mle.update_log(time_tick, stats_tick, save=True)

    best_config = {}
    for i, k in enumerate(param_hyperspace.keys()):
        best_config[k] = hyper_optimizer.get_result().x[i]
    return best_config
