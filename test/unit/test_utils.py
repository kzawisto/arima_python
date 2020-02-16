from nose.tools import *
from arima.utils import *
from hamcrest import *


def test_np_shift():
    assert_that(np_shift([1, 2, 3, 4], 2, -1), only_contains(-1, -1, 1, 2))
    assert_that(np_shift([1, 2, 3, 4], -2, -1), only_contains(3, 4, -1, -1))
    assert_that(np_shift([1, 2, 3, 4], 0, -1), only_contains(1, 2, 3, 4))


def test_shift_stack():
    assert_that(shift_and_stack([1, 2, 3, 4], 2)[
                :, 0], only_contains(0, 1, 2, 3))
    assert_that(shift_and_stack([1, 2, 3, 4], 2)[
                :, 1], only_contains(0, 0, 1, 2))


def test_ols():
    assert_that(ols_0dim([1, 2, 3, 4, 5], [1, 0, 2, 1, 3]),
                close_to(0.47, 0.01))
