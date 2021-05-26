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
    # TODO: Save log with mle experiment?!
    return hyper_optimizer


def main():
    X, y, X_test = get_encoding_data(fmri_dir='./data/participants_data_v2021',
                                     activations_dir='./data/features/alexnet/pca_50',
                                     layer_id='layer_1',
                                     subject_id='sub01', roi_type='V1')

    print(X.shape, y.shape, X_test.shape)
    smbo_config = {"base_estimator": "GP",      # "GP", "RF", "ET", "GBRT"
                   "acq_function": "gp_hedge",  # "LCB", "EI", "PI", "gp_hedge"
                   "n_initial_points": 5,
                   "opt_iters": 20}
    result = run_bayes_opt(smbo_config, lm_params_to_search, X, y)

    # TODO: Periodically - fit best model with full data and predict on test set!
    # Store prediction in submission directory

if __name__ == "__main__":
    #mle = MLExperiment(config_fname="configs/train/base_config.json")
    main()
