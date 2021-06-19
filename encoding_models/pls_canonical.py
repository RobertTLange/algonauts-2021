from sklearn.cross_decomposition import PLSCanonical

# Partial LS Hyperparameters
pls_canonical_params_to_search = {
"integer":
    {"n_components": {"begin": 1, "end": 50, "prior": 'uniform'}},
}


def fit_pls_canonical(model_config, X, y):
    model = PLSCanonical(n_components=model_config["n_components"])
    model.fit(X, y)
    return model


def predict_pls_canonical(model_params, X):
    return model_params.predict(X)
