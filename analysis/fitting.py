# Author: Giraudo Anthony                     
# 19 decembre 2018
# fitting.py


import numpy as np
from scipy.optimize import least_squares


def residual(p, x, y, error):
    r = (y - gaussian(p, x)) / error
    return r


def residual_cut(p, x, y):
    r = y - cutGaussian(p, x)
    return r


def relative_error(y):
    res = 1 / np.sqrt(np.array(y))
    return res


def absolute_error(y):
    res = y * relative_error(y)
    res[y == 0.] = 1. #TODO
    return res


def gaussian(p, x):
    """
    :param p: np.array([high, mean, sd])
    :param x: scalar or np.array
    :return:
    """
    if p[0] == 0:
        return np.zeros(x.shape) # useful ?
    return p[0] * np.exp(-(x-p[1])**2/(2*p[2]**2))


def cutGaussian(p, x):
    """
        :param p: np.array([high, mean, sd])
        :param x: scalar or np.array
        :return:
        """
    res = gaussian(p, x)
    res[res>63.] = 63.
    return res


def p0(x, y):
    high0 = np.amax(y)
    mean0 = np.average(x, weights=y)
    std0 = np.sqrt(np.average((x - mean0) ** 2, weights=y))
    if std0 == 0.:
        std0 = 1.
    p0 = [high0, mean0, std0]
    return p0


def first_fitting(x, y, fitting_type):
    if np.sum(y) < 1e-1:
        return [0, 0, 0]
    if fitting_type == "cut": popt = least_squares(residual_cut, p0(x, y), args=(x, y))
    elif fitting_type == "gaussian": popt = least_squares(residual, p0(x, y), args=(x, y, np.ones(y.shape)))
    else: raise ValueError("Invalid fitting type")
    return popt["x"]


def filtre_artifacts(x, y, fitting_type, param):
    popt = first_fitting(x, y, fitting_type)
    if fitting_type == "gaussian": filtre = abs(y - gaussian(popt, x)) < param
    elif fitting_type == "cut": filtre = ((y != 63.) & (abs(y - gaussian(popt, x)) < param)) | (
        (y == 63.) & (abs(y - cutGaussian(popt, x)) < param))
    else: raise ValueError("Invalid fitting type")
    return filtre, popt


def fitting(x, y):
    if np.sum(y) < 1e-1:
        return [0, 0, 0]
    popt = least_squares(residual, p0(x, y), args=(x, y, absolute_error(y)))
    return popt["x"]


def double_fitting(x_values, y_values, first_fitting_type, param):
    x = np.copy(x_values)
    y = np.copy(y_values)
    filtre, first_popt = filtre_artifacts(x, y, first_fitting_type, param)
    x = x[filtre]
    y = y[filtre]
    x = x[y < 63.]
    y = y[y < 63.]
    return fitting(x, y), filtre, first_popt
