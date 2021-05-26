import os
from mle_toolbox import MLExperiment

from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from utils.helper import get_encoding_data
from encoding_models.trees import fit_gradboost_model, gb_params_to_search
from encoding_models.ols import fit_linear_model, lm_params_to_search
from encoding_models.networks import fit_mlp_model, mlp_params_to_search


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


def run_bayes_opt(mle, params_to_search, X, y):
    """ Simple loop running SMBO + Cross-Validation. """
    param_range = get_hyperspace(params_to_search)
    hyper_optimizer = Optimizer(dimensions=list(param_range.values()),
                                random_state=1,
                                **mle.train_config.smbo_config)
    for t in range(mle.train_config["bo_opt_iters"]):
        proposal = hyper_optimizer.ask()
        model_config = {}
        for i, k in enumerate(param_range.keys()):
            model_config[k] = proposal[i]
        if mle.train_config.encoding_model == "linear_regression":
            cv_score_mean, cv_score_std  = fit_linear_model(model_config, X, y)
        hyper_optimizer.tell(proposal, cv_score_mean)

        time_tick = {"total_bo_iters": t+1}
        stats_tick = {"cv_score_mean": cv_score_mean,
                      "cv_score_std": cv_score_std,
                      "best_bo_score": hyper_optimizer.get_result().fun}
        mle.update_log(time_tick, stats_tick, save=True)
        # TODO: Also store BO parameter!
        # print(hyper_optimizer.get_result().x)
    return hyper_optimizer


def main(mle):
    # Load encoding data - Model features and fMRI targets
    activations_dir = ('./data/features/' +
                       mle.train_config.feature_model +
                       '/pca_' + str(mle.train_config.dim_reduction))
    X, y, X_test = get_encoding_data(fmri_dir=mle.train_config.fmri_dir,
                                     activations_dir=activations_dir,
                                     layer_id=mle.train_config.layer_id,
                                     subject_id=mle.train_config.subject_id,
                                     roi_type=mle.train_config.roi_type)

    # Select parameter space to search over!
    if mle.train_config.encoding_model == "linear_regression":
        params_to_search = lm_params_to_search

    # Run SMBO loop with cross-validation
    result = run_bayes_opt(mle, params_to_search, X, y)

    # TODO: Fit best model with full data and predict on test set!


if __name__ == "__main__":
    mle = MLExperiment(config_fname="configs/train/base_config.json")
    main(mle)
