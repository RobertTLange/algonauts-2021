from sklearn.cross_decomposition import PLSRegression

# Partial LS Hyperparameters
pls_params_to_search = {
"integer":
    {"n_components": {"begin": 1, "end": 50, "prior": 'uniform'}},
}


def fit_partial_ls(model_config, X, y):
    model = PLSRegression(n_components=model_config["n_components"])
    model.fit(X, y)
    return model


def predict_partial_ls(model_params, X):
    return model_params.predict(X)
