import numpy as np
from utils.helper import get_encoding_data
from encoding_models import EncoderFitter
from encoding_models.mlp_networks import build_network
from mle_toolbox import load_result_logs
from mle_toolbox.utils import load_model


def get_expert_encoding_data(roi_paths,
                             fmri_dir='../data/participants_data_v2021',
                             activations_dir='../data/features/alexnet/pca_100',
                             layer_id='layer_1', subject_id='sub01'):

    all_rois = ['LOC', 'FFA', 'STS', 'EBA', 'PPA',
                'V1', 'V2', 'V3', 'V4']
    X_wb_train, y_wb_train, X_wb_test = get_encoding_data(fmri_dir, activations_dir,
                                                          layer_id, subject_id,
                                                          roi_type="WB")

    all_roi_train, all_roi_test = [X_wb_train], [X_wb_test]
    for roi in all_rois:
        encoding_model = roi_paths[roi]["model"]
        X, y, X_test = get_encoding_data(fmri_dir, activations_dir,
                                         layer_id, subject_id, roi)

        meta_log, hyper_log = load_result_logs(roi_paths[roi]["path"])
        hyper_sub = hyper_log[hyper_log.subject_id == subject_id]
        hyper_sub_roi = hyper_sub[hyper_sub.roi_type == roi]
        e_id = hyper_sub_roi.run_id.iloc[0]
        experiment_dir = meta_log[e_id].meta.experiment_dir
        model_ckpt = "../data/" + str(hyper_sub_roi.model_ckpt.iloc[0])
        model_type = hyper_sub_roi.model_type.iloc[0]

        # Need to get network config to 'rebuild' trained predictor
        if encoding_model == "mlp_network":
            config_keys = ['dropout', 'hidden_act', 'learning_rate', 'num_hidden_layers',
                           'num_hidden_units', 'optimizer_class', 'w_decay']
            best_id = np.argmax(meta_log[e_id].stats.best_bo_score.mean)
            best_config = {}
            for c_key in config_keys:
                if type(meta_log[e_id].stats[c_key]) == list:
                    config_value = meta_log[e_id].stats[c_key][0][best_id].decode()
                else:
                    config_value = meta_log[e_id].stats[c_key].mean[best_id]
                best_config[c_key] = config_value
            model = build_network(best_config, X, y)
        else:
            model = None

        fitter = EncoderFitter(encoding_model, None, X, y, X_test)
        model_params = load_model(model_ckpt, model_type, model)
        Xy_train_pred = fitter.predict(model_params, X)
        Xy_test_pred = fitter.predict(model_params, X_test)

        all_roi_train.append(Xy_train_pred)
        all_roi_test.append(Xy_test_pred)

    X_wb_train = np.concatenate(all_roi_train, axis=1)
    X_wb_test = np.concatenate(all_roi_test, axis=1)
    return X_wb_train, y_wb_train, X_wb_test
