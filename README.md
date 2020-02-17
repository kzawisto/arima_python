
# ARIMA model toolbox for Python

## Setup 
Python 3.5 or newer needed
```shell script
pip install git+https://github.com/kzawisto/arima_python
```

## Fitting ARMA model to data, sampling from ARMA process
```python

import arima
#generate some dummy data sampling from process
ar_b, ma_b = [0.3, 0.1], [0.3, 0.2]
x, eps = arima.generate_process(ar_b, ma_b, number=100)
# do actual fit with EM algorithm
ar_est, ma_est = arima.fit_arma(x, ar_terms=2, ma_terms=2)
```

## Testing for stationarity with ADF + bootstrap

```python
import numpy as np
import arima
tester = arima.ADFBootstrap(100, 10000, 3, False, False)
samples = [tester.do_test(np.random.randn(100))['cdf']
           for _ in range(1000)]
assert (np.array(samples) < 0.001).all()
```

## Choosing model using BIC/AIC, forecasting for given params
```python
import arima
import numpy as np
ar_terms, ma_terms = [0.3, 0.4], [0.2, 0.3]
np.random.seed(1)
data,eps = arima.generate_process(ar_terms, ma_terms,10000)
forecast = arima.get_forecast(data, ar_terms, ma_terms)
print("Realized variance", np.var(data-forecast))
results, model=arima.choose_arma_model(data, 5, 5)
assert len(model['ar_params'] == 2)
assert len(model['ma_params'] == 2)
print("Fitted model variance", model['eps_variance'])
```


## Build & test 
```shell script
git clone https://github.com/kzawisto/arima_python
cd arima_python
python setup.py install --force
nosetests test/*/*
```
