

from arima.process import get_forecast, generate_process
import numpy as np


def test_model_forecast():
    ar, ma = [0.3, 0.1], [0.4, 0.3]
    X, eps = generate_process(ar, ma)
    forecast = get_forecast(X, ar, ma)
    assert (np.abs(eps - (X-forecast)) < 1e-6).all()
