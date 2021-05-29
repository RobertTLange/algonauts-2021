from mle_toolbox import MLExperiment
from utils.helper import get_encoding_data
from bayes_opt_search import get_hyperspace, run_bayes_opt
#from perform_encoding import perform_encoding

from encoding_models.trees import fit_gradboost_model, gb_params_to_search
from encoding_models.ols import fit_linear_model, lm_params_to_search
from encoding_models.networks import fit_mlp_model, mlp_params_to_search


def main(mle):
    """ Load data, Bayes opt + CV on model params, Final subm prediction """
    # Load encoding data - Model features and fMRI targets - Layer/Subject
    activations_dir = ('./data/features/' +
                       mle.train_config.feature_model +
                       '/pca_' + str(mle.train_config.dim_reduction))
    X, y, X_test = get_encoding_data(fmri_dir=mle.train_config.fmri_dir,
                                     activations_dir=activations_dir,
                                     layer_id=mle.train_config.layer_id,
                                     subject_id=mle.train_config.subject_id,
                                     roi_type=mle.train_config.roi_type)

    # Get params, update keys to track and run SMBO loop with cross-validation
    if mle.net_config.encoding_model == "linear_regression":
        params_to_search = lm_params_to_search
    param_hyperspace = get_hyperspace(params_to_search)
    mle.log.extend_tracking(list(param_hyperspace.keys()))

    best_params = run_bayes_opt(mle, param_hyperspace, X, y)

    # Fit best model with full data, predict on test set!
    # y_pred = perform_test_encoding(mle, best_params, X, y, X_test)

    # Store best models predictions and model itself


if __name__ == "__main__":
    mle = MLExperiment(config_fname="configs/train/base_config.json")
    main(mle)
