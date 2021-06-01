from sklearn.linear_model import ElasticNet

# Elastic Net Regression Hyperparameters
elastic_params_to_search = {
"real":
    {"alpha": {"begin": 1e-5, "end": 100, "prior": 'uniform'},
     "l1_ratio": {"begin": 0, "end": 1, "prior": 'uniform'}},
}

def fit_elastic_net(model_config, X, y):
    # alpha = model_config["l1_lambda"] + model_config["l2_lambda"]
    # l1_ratio = model_config["l1_lambda"] / (model_config["l1_lambda"] +
    #                                         model_config["l2_lambda"])
    model = ElasticNet(model_config["alpha"], model_config["l1_ratio"])
    model.fit(X, y)
    return model

def predict_elastic_net(model_params, X):
    return model_params.predict(X)
