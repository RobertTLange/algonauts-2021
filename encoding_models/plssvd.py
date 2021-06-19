from sklearn.cross_decomposition import PLSSVD

# Partial LS Hyperparameters
plssvd_params_to_search = {
"integer":
    {"n_components": {"begin": 1, "end": 50, "prior": 'uniform'}},
}


def fit_plssvd(model_config, X, y):
    model = PLSSVD(n_components=model_config["n_components"])
    model.fit(X, y)
    return model


def predict_plssvd(model_params, X):
    return model_params.predict(X)
