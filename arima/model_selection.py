
import numpy as np


def aic(var_eps, deg_of_freedom, n_points):
    return 2 * deg_of_freedom - 2*np.log(var_eps)


def bic(var_eps, deg_of_freedom, n_points):
    pass
