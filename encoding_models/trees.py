from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


# Sklearn GradientBoostingRegressor Hyperparameters
gb_params_to_search = {
"real":
    {"alpha": {"begin": 0.01, "end": 0.999, "prior": 'log-uniform'},
     "learning_rate": {"begin": 0.01, "end": 0.99, "prior": 'log-uniform'}
     },
"integer":
    {'max_depth': {"begin": 1, "end": 50, "prior": 'uniform'},
     'n_estimators': {"begin": 50, "end": 500, "prior": 'uniform'},
     }
}

gb_default_params = {"num_threads": 5}

def fit_gradboost_model(model_config, X, y):
    model = GradientBoostingRegressor(loss='ls',
                                      min_samples_leaf=9,
                                      min_samples_split=9,
                                      **model_config)
    wrapper = MultiOutputRegressor(model,
                                   n_jobs=gb_default_params["num_threads"])
    wrapper.fit(X, y)
    return wrapper


def predict_gradboost_model(model_params, X):
    return model_params.predict(X)
