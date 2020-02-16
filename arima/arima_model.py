

from arima.utils import *
import warnings


def fit_arma(X, ar_terms=2, ma_terms=2, eps_limit=1e-3, max_steps=30):
    arg = shift_and_stack(X, ar_terms)
    initial_beta = ols(arg, X)
    eps = X-np.sum(initial_beta*arg, axis=1)
    q = np.mean(eps**2)
    for i in range(max_steps):
        arg = shift_and_stack(X, ar_terms)
        arg2 = shift_and_stack(eps, ma_terms)

        arg_all = np.concatenate([arg, arg2], axis=1)
        beta = ols(arg_all, X)
        eps = X-np.sum(beta*arg_all, axis=1)
        # print(beta, np.mean(eps**2)-q, np.mean(eps**2))
        if eps_limit > np.abs(np.mean(eps ** 2) - q):
            break
        q = np.mean(eps**2)
        if i == max_steps - 1:
            warnings.warn("Failed to converge within " + str(max_steps))
        # if eps
    return beta[:ar_terms][::-1], beta[ar_terms:][::-1]
