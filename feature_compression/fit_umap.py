import umap


def fit_trafo_umap(x_train, x_test, umap_params):
    fit_params ={"n_neighbors": 100,
                  "min_dist": 0.5,
                  "n_components": umap_params["n_components"],
                  "metric":'euclidean',
                  "n_jobs": 10}

    dim_red = umap.UMAP(**fit_params)
    trafo = dim_red.fit(x_train)
    umap_info = {}
    x_train_trafo = trafo.transform(x_train)
    x_test_trafo = trafo.transform(x_test)
    return x_train_trafo, x_test_trafo, umap_info
