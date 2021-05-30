from encoding_models.ols import (fit_linear_model,
                                 predict_linear_model,
                                 lm_params_to_search)
# from encoding_models.trees import (fit_gradboost_model,
#                                    predict_gradboost_model,
#                                    gb_params_to_search)
from encoding_models.mlp_networks import (fit_mlp_model,
                                          predict_mlp_model,
                                          mlp_params_to_search)
import numpy as np
from sklearn.model_selection import RepeatedKFold
from utils.evaluate import evaluation_metrics


def get_model_hyperparams(model_name):
    if model_name == "linear_regression":
        return lm_params_to_search
    elif model_name == "mlp_network":
        return mlp_params_to_search


class EncoderFitter(object):
    def __init__(self, model_name, num_cv_folds, X, y, X_test):
        """ Wrapper Class for CV fitting and predicting test data. """
        self.model_name = model_name
        self.num_cv_folds = num_cv_folds
        self.X = X
        self.y = y
        self.X_test = X_test

    @property
    def hyperparams(self):
        """ Considered hyperparameters in search/BO. """
        return get_model_hyperparams(self.model_name)

    def fit(self, model_config, X_train, y_train, X_val, y_val):
        """ Fit and evaluate model once. """
        if self.model_name == "linear_regression":
            model_params = fit_linear_model(model_config, X_train, y_train)
            y_pred = predict_linear_model(model_params, X_val)
        elif self.model_name == "mlp_network":
            model_params = fit_mlp_model(model_config, X_train, y_train)
            y_pred = predict_mlp_model(model_params, X_val)
        mse, mae, corr = evaluation_metrics(y_val, y_pred)
        return mse, mae, corr

    def cv_fit(self, model_config):
        """ Run cross-validation fitting and evaluation. """
        cv = RepeatedKFold(n_splits=self.num_cv_folds,
                           n_repeats=1, random_state=1)
        mse_scores, mae_scores, corr_scores = [], [], []
        for train_index, val_index in cv.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]
            corr, mse, mae = self.fit(model_config,
                                      X_train, y_train,
                                      X_val, y_val)
            mse_scores.append(mse)
            mae_scores.append(mae)
            corr_scores.append(corr)
        cv_scores_mean = {"mse_mean": np.mean(mse_scores),
                          "mae_mean": np.mean(mae_scores),
                          "corr_mean": np.mean(corr_scores)}
        cv_scores_std = {"mse_std": np.std(mse_scores),
                         "mae_std": np.std(mae_scores),
                         "corr_std": np.std(corr_scores)}
        return cv_scores_mean, cv_scores_std

    def predict_on_test(self, model_config):
        """ Fit model on entire data and return predictions on test. """
        if self.model_name == "linear_regression":
            model_params = fit_linear_model(model_config, self.X, self.y)
            y_pred = predict_linear_model(model_params, self.X_test)
        return y_pred
