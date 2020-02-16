
import numpy as np

import scipy.stats as sst
from arima.utils import *


def wald_stat(X1, eps, beta):
    # assert X1.shape[0] < X1.shape[1], "X dim: " + str(X1.shape) + " should be rotated"
    if X1.shape[0] > X1.shape[1]:
        X1 = X1.T
    beta_covariance = np.var(eps) * np.linalg.pinv(
        np.dot(X1, X1.T)
    )
    wald_stat = np.dot(
        np.dot(beta.reshape(1, -1), np.linalg.pinv(beta_covariance)), beta)
    return float(wald_stat)


def augmented_dickey_fuller_fit(X, p=5, constant_term=False, trend_term=True):
    """
    Fit regression of form:
    Diff X_i = yX_(i-1) + a_1 Diff X_(i-1) +... + a_p X_(i-p)   + c + di
    where
    Diff X_i = X_i - X_(i-1)
    :param X: series to fit
    :param p:
    :param constant_term: if False we set c =0
    :param trend_term: if False we set d = 0
    :return: [y, a_1...a_p, c, d]
    """
    Xdiff = np.zeros_like(X)
    Xdiff[1:] = np.diff(X)
    Xlagged = shift_and_stack(X, 1)
    if p != 0:
        df = np.concatenate([Xlagged, shift_and_stack(Xdiff, p)], axis=1)
    else:
        df = Xlagged
    if constant_term:
        df = np.concatenate([df, np.ones_like(Xlagged)], axis=1)

    if trend_term:
        index = np.arange(len(X)).reshape(-1, 1)
        df = np.concatenate([df, index], axis=1)
    betas = ols(df, Xdiff)
    return betas


class ADFBootstrap(object):
    def __init__(self, N, bootstrap_steps=1000, p=5, constant_term=False, trend_term=False, func=None):
        if func == None:
            def func(x): return np.cumsum(np.random.randn(x))
        self.p = p
        self.constant_term = constant_term
        self.trend_term = trend_term
        self.N = N

        def sample():
            X = func(self.N)
            return augmented_dickey_fuller_fit(X, self.p, self.constant_term, self.trend_term)[0]
        self.samples = np.array(
            sorted(([sample() for _ in range(bootstrap_steps)])))

    def cdf(self, adf_value):

        return (np.searchsorted(self.samples, adf_value) + 0.5) / len(self.samples)

    def do_test(self, X, p_crit=0.01):
        assert len(X) == self.N, "len(X) != " + str(self.N)
        beta = augmented_dickey_fuller_fit(
            X, self.p, self.constant_term, self.trend_term)[0]
        beta_cdf = self.cdf(beta)
        return {
            "y": beta,
            'cdf': beta_cdf,
            'result': "no_root" if beta_cdf < p_crit else "unit_root"
        }
