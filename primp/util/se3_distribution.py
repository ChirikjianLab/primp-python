from primp.util.se3_util import get_exp_mapping, get_exp_coord

import numpy as np
import warnings


def get_mean(g, group_name="SE"):
    """
    Compute mean of a set of SE(3)
    :param g: The set of SE(3) elements
    :param group_name: Name of the group
    :return: Mean of the set
    """
    num = len(g)

    # Mean
    # Initialization
    mu = np.identity(4)
    for i in range(5):
        mu_log = np.zeros((6, ))
        for j in range(num):
            mu_log += get_exp_coord(g[j], group_name)
        mu = mu @ get_exp_mapping(1.0/num * mu_log)

    # Iterative process to compute mean
    mu_log = np.ones((6, ))
    max_num = 10
    tol = 1e-5
    count = 1
    while np.linalg.norm(mu_log) >= tol and count <= max_num:
        mu_log = np.zeros((6, ))
        for i in range(num):
            d_g_log = get_exp_coord(np.linalg.inv(mu) @ g[i], group_name)
            mu_log = mu_log + d_g_log
        mu = mu @ get_exp_mapping(1.0/num * mu_log, group_name)
        count += 1

    if count > max_num:
        warnings.warn("Cannot obtain correct mean pose, error: " + str(np.linalg.norm(mu_log)))
        flag = False
        return mu, flag

    flag = True
    return mu, flag


def get_covariance(g, mu=None, group_name="SE"):
    """
    Compute covariance of a set of SE(3)
    :param g: The set of SE(3) elements
    :param mu: Computed mean or None
    :param group_name: Name of the group
    :return: Covariance of the set
    """
    if mu is None:
        mu, flag = get_mean(g)

    num = len(g)

    # Covariance
    sigma = np.zeros((6, 6))
    for i in range(num):
        y_i = np.array([get_exp_coord(np.linalg.inv(mu) @ g[i], group_name)])
        sigma += y_i.T @ y_i  # y_i is a row vector
    sigma *= 1.0/num

    return sigma