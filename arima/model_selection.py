
import numpy as np

from arima import get_forecast
from arima.arima_model import fit_arma


def aic(var_eps, deg_of_freedom, n_points):
    return 2 * deg_of_freedom + 2*n_points*np.log(var_eps)


def bic(var_eps, deg_of_freedom, n_points):
    return np.log(n_points)* deg_of_freedom +2 *n_points* np.log(var_eps)


def choose_arma_model(data, max_ar_lags=3, max_ma_lags=3, criterion='bic'):
    results = {}
    if isinstance(criterion,str):
        criterion = {
            'bic': bic, 'aic': aic
        }[criterion]
    for ar_lag in range(max_ar_lags+1):
        for ma_lag in range(max_ma_lags+1):

            ar_params, ma_params= fit_arma(data, ar_lag, ma_lag )
            forecast = get_forecast(data, ar_params, ma_params)
            results[(ar_lag,ma_lag)]={
                'ar_params': ar_params,
                'ma_params':ma_params,
                'criterion':criterion(np.var(data - forecast), ar_lag+ma_lag, len(data)),
                'eps_variance': np.var(data - forecast)
            }
    # results = sorted(results, key = lambda x: x['criterion'])
    return results, sorted([r for r in results.values() if not np.isnan(r['criterion'])], key=lambda x: x['criterion'])[0]


