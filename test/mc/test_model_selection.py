from arima import generate_process, get_forecast
from arima.model_selection import choose_arma_model
import numpy as np,pandas as pd


def test_arma_model_selection_reproduces_original():
    ar_terms, ma_terms = [0.3, 0.4], [0.2, 0.3]
    np.random.seed(1)
    data,eps = generate_process(ar_terms, ma_terms,10000)
    forecast = get_forecast(data, ar_terms, ma_terms)
    print("Variance of data", np.var(data))
    results, model=choose_arma_model(data, 8, 8)
    print(model)
    print(results)
    assert len(model['ar_params'] == 2)
    assert len(model['ma_params'] == 2)
    assert np.var(data-forecast) > model["eps_variance"]

