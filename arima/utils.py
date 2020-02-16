import numpy as np


def ols(X, Y):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))


def np_shift(array, lags, fill=0):
    result = np.zeros_like(array)
    if lags > 0:
        result[lags:] = array[:-lags]
        result[:lags] = fill
    if lags < 0:
        result[:-lags] = array[lags:]
        result[-lags:] = fill
    if lags == 0:
        result = np.array(array)
    return result


def shift_and_stack(X, num_lags):
    return np.stack([np_shift(X, i) for i in range(1, num_lags + 1)]).T


def ols_0dim(X, Y):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    return np.dot(X.T, Y) / np.dot(X.T, X)
