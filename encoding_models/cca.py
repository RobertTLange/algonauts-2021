from sklearn.cross_decomposition import CCA

# Partial LS Hyperparameters
cca_params_to_search = {
"integer":
    {"n_components": {"begin": 1, "end": 50, "prior": 'uniform'}},
}


def fit_cca(model_config, X, y):
    model = CCA(n_components=model_config["n_components"])
    model.fit(X, y)
    return model


def predict_cca(model_params, X):
    return model_params.predict(X)
