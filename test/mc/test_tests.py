from arima.stat_test import augmented_dickey_fuller_fit, wald_stat, ols, ADFBootstrap

import numpy as np
import scipy.stats as sst

from hamcrest import *


def test_simple_adf():
    def sample(N, p=5):
        X = np.cumsum(np.random.randn(N))
        return augmented_dickey_fuller_fit(X, p, False, False)[0]

    np.random.seed(1)
    samples = [sample(1000) for _ in range(1000)]
    assert_that(np.mean(samples), close_to(0, 0.01))
    assert_that(np.std(samples), close_to(0, 003.0002))
    assert_that(sst.skew(samples), close_to(-2, 0.01))


def test_adf_w_const():

    np.random.seed(7)

    def sample(N, p=5):
        X = np.cumsum(np.random.randn(N))
        return augmented_dickey_fuller_fit(X, p, True, False)[0]

    np.random.seed(1)
    samples = [sample(1000) for _ in range(1000)]
    assert_that(np.mean(samples), close_to(0, 0.01))
    assert_that(np.std(samples), close_to(0.005, 0.001))
    assert_that(sst.skew(samples), close_to(-1.4, 0.01))


def test_adf_w_const_and_trend():

    np.random.seed(7)

    def sample(N, p=5):
        X = np.cumsum(np.random.randn(N))
        return augmented_dickey_fuller_fit(X, p, True, True)[0]

    np.random.seed(1)
    samples = [sample(1000) for _ in range(1000)]
    assert_that(np.mean(samples), close_to(0, 0.02))
    assert_that(np.std(samples), close_to(0.0058, 0.0001))
    assert_that(sst.skew(samples), close_to(-0.94, 0.01))


def test_wald_ols():
    # Should be chi2(N) if nothing here

    np.random.seed(7)

    def sample():
        X = np.random.randn(1000, 3)
        y = np.random.randn(1000)
        betas = ols(X, y)
        # print(np.sum(betas * X, axis=1))
        eps = y - np.sum(betas*X, axis=1)
        return wald_stat(X, eps, betas)
    samples = [sample() for _ in range(10000)]
    assert sst.kstest(samples, sst.chi2(
        3).cdf).pvalue > 0.05, "Sample distribution is not chi2(3)"


def test_adf_bootstrap_on_normal():

    np.random.seed(2)
    tester = ADFBootstrap(100, 10000, 3, False, False)
    samples = [tester.do_test(np.random.randn(100))['cdf']
               for _ in range(1000)]
    assert (np.array(samples) < 0.001).all()


def test_adf_bootstrap_on_wiener_random_walk():
    np.random.seed(3)
    tester = ADFBootstrap(100, 10000, 3, False, False)
    samples = [tester.do_test(np.cumsum(np.random.randn(100)))[
        'cdf'] for _ in range(1000)]
    assert (sst.kstest(samples, sst.uniform.cdf).pvalue > 0.05)
