import numba
import numpy as np

#
# def generate_process(ar_terms, ma_terms, number=1000, eps=None):
#     if len(ar_terms) == 0:
#         ar_terms = np.array([0])
#     if len(ma_terms) == 0:
#         ma_terms = np.array([0])
#
#     terms = max(len(ar_terms), len(ma_terms))
#     if eps is None:
#         eps = np.random.randn(number)
#     X = np.zeros_like(eps)
#     X[:terms] = eps[:terms]
#     for i in range(terms, len(X)):
#         X[i] = eps[i] + np.sum(eps[i - len(ma_terms):i] * ma_terms) + \
#             np.sum(X[i - len(ar_terms):i] * ar_terms)
#     return X, eps


def decorator(x): return x


try:
    import numba
    decorator = numba.jit
except ImportError as e:
    import warnings
    warnings.warn(
        "Failed to import numba - generation and forecast will be 20x slower. Please install numba (pip install numba). Error:", str(e))


@decorator
def generate_process(ar_terms, ma_terms, number=1000, eps=None):

    if len(ar_terms) == 0:
        ar_terms = np.array([0])
    if len(ma_terms) == 0:
        ma_terms = np.array([0])
    len_ma_terms = len(ma_terms)
    len_ar_terms = len(ar_terms)
    terms = max(len(ar_terms), len(ma_terms))
    if eps is None:
        eps = np.random.randn(number)
    X = np.zeros_like(eps)
    X[:terms] = eps[:terms]
    for i in range(terms, len(X)):
        # + np.sum(eps[i - len(ma_terms):i] * ma_terms) + np.sum(X[i - len(ar_terms):i] * ar_terms)
        X[i] = eps[i]
        for j in range(len_ar_terms):
            X[i] += ar_terms[j] * X[i - len(ar_terms)+j]
        for j in range(len_ma_terms):
            X[i] += ma_terms[j] * eps[i - len(ma_terms)+j]
    return X, eps


@decorator
def get_forecast(X, ar_terms, ma_terms):
    eps = np.zeros_like(X)
    forecast = np.zeros_like(X)

    if len(ar_terms) == 0:
        ar_terms = np.array([0])
    if len(ma_terms) == 0:
        ma_terms = np.array([0])
    len_ma_terms = len(ma_terms)
    len_ar_terms = len(ar_terms)
    terms = max(len(ar_terms), len(ma_terms))

    eps[:terms] = X[:terms]
    for i in range(terms, len(X)):
        for j in range(len_ar_terms):
            forecast[i] += ar_terms[j] * X[i - len(ar_terms) + j]
        for j in range(len_ma_terms):
            forecast[i] += ma_terms[j] * eps[i - len(ma_terms) + j]
        eps[i] = X[i] - forecast[i]
    return forecast
