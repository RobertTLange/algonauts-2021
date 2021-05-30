import torch
import numpy as np


def vectorized_correlation(x,y):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    dim = 0
    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, keepdims=True) + 1e-8
    y_std = y.std(axis=dim, keepdims=True) + 1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr[0]


def evaluation_metrics(y, y_pred):
    corr = vectorized_correlation(y, y_pred)
    mse = np.mean((y-y_pred)**2)
    mae = np.mean(np.abs(y-y_pred))
    return corr, mse, mae
