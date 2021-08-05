import os
import numpy as np
from mle_toolbox import MLExperiment
from mle_toolbox.utils import print_framed
from utils.helper import get_encoding_data
from utils.evaluate import evaluation_metrics
from sklearn.metrics import r2_score
from autosklearn.regression import AutoSklearnRegressor
# Check docs/AutoML params here: https://automl.github.io/auto-sklearn/master/api.html#regression


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
    mle.log.extend_tracking(["layer_id"])

    # Loop over all layers to consider and run BO search for each one!
    for i, layer_id in enumerate(feature_layers_to_consider):
        print_framed(f"Start Layer {layer_id}"
                     + f" - {mle.train_config.subject_id}"
                     + f" - {mle.train_config.roi_type}")
        # Load encoding data - Model features and fMRI targets - Layer/Subject
        X, y, X_test = get_encoding_data(fmri_dir=mle.train_config.fmri_dir,
                                         activations_dir=activations_dir,
                                         layer_id=layer_id,
                                         subject_id=mle.train_config.subject_id,
                                         roi_type=mle.train_config.roi_type)
        # print(os.path.join(mle.log_dir, "autosklearn"))
        # Instatiate Auto Sklearn and Run
        dataname = f'{mle.train_config.subject_id}_{mle.train_config.roi_type}_{layer_id}'
        automl = AutoSklearnRegressor(
                        **mle.train_config.autosklearn_config,
                        tmp_folder=os.path.join(mle.log_dir, "autosklearn", dataname),
                        n_jobs=-1)
        automl.fit(X, y, dataset_name=dataname)

        y_train_pred = automl.predict(X)
        corr, mse, mae = evaluation_metrics(y, y_train_pred)
        r2_stats = r2_score(y, y_train_pred)

        time_tick = {"layer": i+1}
        stats_tick = {"layer_id": layer_id,
                      "mse_mean": mse,
                      "mae_mean": mae,
                      "corr_mean": corr,
                      "r2_score": r2_stats}
        model = {"ensemble": automl.show_models()}
        extra = automl.cv_results_['params'][np.argmax(automl.cv_results_['mean_test_score'])]
        mle.update_log(time_tick, stats_tick, model, extra_obj=extra, save=True)

        # After done with BO for layer features - save current best predictions
        print_framed(f"Save model predictions {layer_id}"
                     + f" - {mle.train_config.subject_id}"
                     + f" - {mle.train_config.roi_type}")

        y_pred = automl.predict(X_test)
        # Store best models predictions and model itself
        pred_fname = (mle.train_config.subject_id + "_" +
                      mle.train_config.roi_type + "_" +
                      str(layer_id) + "_test.npy")
        mle.log.save_extra(y_pred, pred_fname)


if __name__ == "__main__":
    mle = MLExperiment()
    main(mle)
