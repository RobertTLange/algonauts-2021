import numpy as np
from sklearn import manifold


def fit_trafo_mds(x_train, x_test, mds_params):
    # Full vs incremental (depending on pca dim and sampling rate)
    mds = manifold.MDS(**mds_params, max_iter=5000, n_jobs=20)
    x_concat = np.concatenate([x_train, x_test], axis=0)
    x_trafo = mds.fit_transform(x_concat)
    mds_info = {"stress": mds.stress_,
                "n_iter": mds.n_iter_}
    x_train_trafo = x_trafo[:1000]
    x_test_trafo = x_trafo[1000:]
    return x_train_trafo, x_test_trafo, mds_info
