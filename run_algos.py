from mle_toolbox import MLExperiment
from utils.helper import get_encoding_data
from bayes_opt_search import get_hyperspace, run_bayes_opt
from encoding_models import EncoderFitter


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

    # Get params, update tracked keys + run SMBO loop with CV
    fitter = EncoderFitter(mle.net_config.encoding_model,
                           mle.train_config.num_cv_folds,
                           X, y, X_test)
    param_hyperspace = get_hyperspace(fitter.hyperparams)
    mle.log.extend_tracking(list(param_hyperspace.keys()))
    best_config = run_bayes_opt(mle, fitter, param_hyperspace)

    # Fit best model with full data, predict on test set!
    y_pred = fitter.predict_on_test(best_config)

    # Store best models predictions and model itself
    pred_fname = "to_define"
    mle.log.save_to_extra_dir(y_pred, pred_fname)

if __name__ == "__main__":
    mle = MLExperiment(config_fname="configs/train/base_config.json")
    main(mle)
