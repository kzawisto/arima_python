

from arima.process import *
from arima.arima_model import *

import time


def test_fit_to_process():
    # this to jit compile function before time measurement
    generate_process([0.3, 0.2, -0.3], [0.2, 0.4], number=1000)
    t = time.time()

    ar_b, ma_b = [0.3, 0.2, -0.3], [0.2, 0.4]

    def fit_and_get_distance(_ar_b, _ma_b, N, reps):
        result = []
        for i in range(reps):
            x, eps = generate_process(_ar_b, _ma_b, number=N)
            ar_est, ma_est = fit_arma(x, 3, 2)
            result.append((np.sum((ar_est - _ar_b) ** 2) +
                           np.sum((ma_est - _ma_b) ** 2)) / 5)
        return np.array(result)

    np.random.seed(1)
    results = fit_and_get_distance(ar_b, ma_b, 2000, 30)
    assert np.quantile(results, 0.9) < 0.05
    print("Time", time.time()-t)
