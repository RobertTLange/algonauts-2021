from encoding_models.ols import (fit_linear_model,
                                 predict_linear_model,
                                 lm_params_to_search)
from encoding_models.trees import (fit_gradboost_model,
                                   predict_gradboost_model,
                                   gb_params_to_search)
from encoding_models.mlp_networks import (fit_mlp_model,
                                          predict_mlp_model,
                                          mlp_params_to_search)
from encoding_models.elastic_net import (fit_elastic_net,
                                         predict_elastic_net,
                                         elastic_params_to_search)
from encoding_models.partial_ls import (fit_partial_ls,
                                        predict_partial_ls,
                                        pls_params_to_search)
from encoding_models.cca import (fit_cca,
                                 predict_cca,
                                 cca_params_to_search)
from encoding_models.pls_canonical import (fit_pls_canonical,
                                    predict_pls_canonical,
                                    pls_canonical_params_to_search)
import numpy as np
from sklearn.model_selection import RepeatedKFold
from utils.evaluate import evaluation_metrics


def get_model_hyperparams(model_name):
    if model_name == "linear_regression":
        return lm_params_to_search
    elif model_name == "mlp_network":
        return mlp_params_to_search
    elif model_name == "elastic_net":
        return elastic_params_to_search
    elif model_name == "gradboost":
        return gb_params_to_search
    elif model_name == "partial_ls":
        return pls_params_to_search
    elif model_name == "cca":
        return cca_params_to_search
    elif model_name == "pls_canonical":
        return pls_canonical_params_to_search


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
        elif self.model_name == "elastic_net":
            model_params = fit_elastic_net(model_config, X_train, y_train)
            y_pred = predict_elastic_net(model_params, X_val)
        elif self.model_name == "gradboost":
            model_params = fit_gradboost_model(model_config, X_train, y_train)
            y_pred = predict_gradboost_model(model_params, X_val)
        elif self.model_name == "partial_ls":
            model_params = fit_partial_ls(model_config, X_train, y_train)
            y_pred = predict_partial_ls(model_params, X_val)
        elif self.model_name == "cca":
            model_params = fit_cca(model_config, X_train, y_train)
            y_pred = predict_cca(model_params, X_val)
        elif self.model_name == "pls_canonical":
            model_params = fit_pls_canonical(model_config, X_train, y_train)
            y_pred = predict_pls_canonical(model_params, X_val)
        corr, mse, mae = evaluation_metrics(y_val, y_pred)
        return corr, mse, mae

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
        elif self.model_name == "mlp_network":
            model_params = fit_mlp_model(model_config, self.X, self.y)
        elif self.model_name == "elastic_net":
            model_params = fit_elastic_net(model_config, self.X, self.y)
        elif self.model_name == "gradboost":
            model_params = fit_gradboost_model(model_config, self.X, self.y)
        elif self.model_name == "partial_ls":
            model_params = fit_partial_ls(model_config, self.X, self.y)
        elif self.model_name == "cca":
            model_params = fit_cca(model_config, self.X, self.y)
        elif self.model_name == "pls_canonical":
            model_params = fit_pls_canonical(model_config, self.X, self.y)
        y_pred = self.predict(model_params, self.X_test)
        return model_params, y_pred

    def predict(self, model_params, X):
        """ Predict using model config. """
        if self.model_name == "linear_regression":
            y_pred = predict_linear_model(model_params, X)
        elif self.model_name == "mlp_network":
            y_pred = predict_mlp_model(model_params, X)
        elif self.model_name == "elastic_net":
            y_pred = predict_elastic_net(model_params, X)
        elif self.model_name == "gradboost":
            y_pred = predict_gradboost_model(model_params, X)
        elif self.model_name == "partial_ls":
            y_pred = predict_partial_ls(model_params, X)
        elif self.model_name == "cca":
            y_pred = predict_cca(model_params, X)
        elif self.model_name == "pls_canonical":
            y_pred = predict_pls_canonical(model_params, X)
        return y_pred
