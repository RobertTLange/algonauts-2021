from sklearn.linear_model import OrthogonalMatchingPursuit

# Partial LS Hyperparameters
omp_params_to_search = {
"integer":
    {"n_nonzero_coefs": {"begin": 1, "end": 50, "prior": 'uniform'}},
}


def fit_omp(model_config, X, y):
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=model_config["n_nonzero_coefs"])
    model.fit(X, y)
    return model


def predict_omp(model_params, X):
    return model_params.predict(X)
