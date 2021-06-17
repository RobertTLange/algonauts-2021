import umap


def fit_trafo_umap(x_train, x_test, umap_params):
    dim_red = umap.UMAP(**umap_params)
    trafo = dim_red.fit(x_train)
    umap_info = {}
    x_train_trafo = trafo.transform(x_train)
    x_test_trafo = trafo.transform(x_test)
    return x_train_trafo, x_test_trafo, umap_info
