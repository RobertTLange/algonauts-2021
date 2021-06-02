from sklearn.linear_model import ElasticNet

# Elastic Net Regression Hyperparameters
elastic_params_to_search = {
"real":
    {"alpha": {"begin": 0.01, "end": 0.25, "prior": 'uniform'},
     "l1_ratio": {"begin": 0.01, "end": 1, "prior": 'uniform'}},
}


def fit_elastic_net(model_config, X, y):
    # alpha = model_config["l1_lambda"] + model_config["l2_lambda"]
    # l1_ratio = model_config["l1_lambda"] / (model_config["l1_lambda"] +
    #                                         model_config["l2_lambda"])
    model = ElasticNet(alpha=model_config["alpha"],
                       l1_ratio=model_config["l1_ratio"],
                       max_iter=50000)
    model.fit(X, y)
    return model


def predict_elastic_net(model_params, X):
    return model_params.predict(X)
