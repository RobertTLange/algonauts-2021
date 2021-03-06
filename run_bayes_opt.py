import os
import numpy as np
from mle_toolbox import MLExperiment
from mle_toolbox.utils import print_framed
from utils.helper import get_encoding_data
from utils.bayes_opt_search import get_hyperspace, run_bayes_opt
from encoding_models import EncoderFitter, get_model_hyperparams


def main(mle: MLExperiment):
    """ Load data, Bayes opt + CV on model params, Final subm prediction """
    # Set directory for all stored model features to train on
    activations_dir = ('./data/features/' +
                       mle.train_config.feature_model +
                       '/' + mle.train_config.filter_type +
                       '/' + 'sr_' + str(mle.train_config.sampling_rate) +
                       '/' + mle.train_config.dim_reduction)

    act_files = [f for f in os.listdir(activations_dir) if
                 os.path.isfile(os.path.join(activations_dir, f))]
    max_layer_id = max([int(f.split("_")[-1].split('.')[0])
                        for f in act_files if f.endswith('.npy')])
    feature_layers_to_consider = [f"layer_{i}" for i in
                                  range(1, max_layer_id+1)]

    # Get model params to optimize over, update tracked keys
    model_hyperparams = get_model_hyperparams(mle.model_config.encoding_model)
    param_hyperspace = get_hyperspace(model_hyperparams)
    mle.log.extend_tracking(list(param_hyperspace.keys()) + ["layer_id"])

    # Loop over all layers to consider and run BO search for each one!
    layer_perf_tracker, best_layer, best_perf = {}, None, np.inf
    for layer_id in feature_layers_to_consider:
        print_framed(f"Start Layer {layer_id}"
                     + f" - {mle.train_config.subject_id}"
                     + f" - {mle.train_config.roi_type}")
        # Load encoding data - Model features and fMRI targets - Layer/Subject
        X, y, X_test = get_encoding_data(fmri_dir=mle.train_config.fmri_dir,
                                         activations_dir=activations_dir,
                                         layer_id=layer_id,
                                         subject_id=mle.train_config.subject_id,
                                         roi_type=mle.train_config.roi_type)

        # Instatiate CV-fitter class + run SMBO loop with CV
        fitter = EncoderFitter(mle.model_config.encoding_model,
                               mle.train_config.num_cv_folds,
                               X, y, X_test)
        best_l_score, best_l_config = run_bayes_opt(mle, fitter,
                                                    param_hyperspace,
                                                    layer_id)
        layer_perf_tracker[layer_id] = {"score": best_l_score,
                                        "config": best_l_config}
        # Track best layer + model config
        if best_l_score < best_perf:
            best_layer = layer_id
            best_perf = best_l_score

        # After done with BO for layer features - save current best predictions
        print_framed(f"Save model predictions {layer_id}"
                     + f" - {mle.train_config.subject_id}"
                     + f" - {mle.train_config.roi_type}")
        # Fit best model full best layer data, predict on test set + save model!
        best_config = layer_perf_tracker[best_layer]["config"]
        X, y, X_test = get_encoding_data(fmri_dir=mle.train_config.fmri_dir,
                                         activations_dir=activations_dir,
                                         layer_id=best_layer,
                                         subject_id=mle.train_config.subject_id,
                                         roi_type=mle.train_config.roi_type)
        fitter = EncoderFitter(mle.model_config.encoding_model,
                               mle.train_config.num_cv_folds,
                               X, y, X_test)
        model_params, y_pred = fitter.predict_on_test(best_config)
        mle.log.save_model(model_params)

    # Store best models predictions and model itself
    pred_fname = (mle.train_config.subject_id + "_" +
                  mle.train_config.roi_type + "_" +
                  best_layer + "_test.npy")
    mle.log.save_extra(y_pred, pred_fname)


if __name__ == "__main__":
    mle = MLExperiment()
    main(mle)
