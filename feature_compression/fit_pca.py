from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA


def fit_trafo_pca(x_train, x_test, pca_params):
    # Full vs incremental (depending on pca dim and sampling rate)
    pca = PCA(**pca_params)#, batch_size=20)
    pca.fit(x_train)
    #explained_variance.append(pca.explained_variance_ratio_.cumsum()[-1])

    x_train_trafo = pca.transform(x_train)
    x_test_trafo = pca.transform(x_test)
    return x_train_trafo, x_test_trafo
