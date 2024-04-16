#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface for ProMP in learning trajectory distribution from demonstration

@author: Sipu Ruan, 2022
"""

from movement_primitives.promp import ProMP
import numpy as np


def promp_learn(T, X, n_weights_per_dim=30):
    """
    Learning process from ProMP

    :param T: Time sequence
    :param X: Data points in Euclidean space
    :param n_weights_per_dim: Number of weights per dimension
    :return: promp: Class object for ProMP
    :return: mean: Mean trajectory
    :return: std: An array of standard deviation
    :return: cov: An array of covariance
    """
    promp = ProMP(n_weights_per_dim=n_weights_per_dim, n_dims=X.shape[2])
    promp.imitate(T, X)
    mean = promp.mean_trajectory(T[0])
    cov = promp.cov_trajectory(T[0])
    std = np.sqrt(promp.var_trajectory(T[0]))

    return promp, mean, std, cov


# ProMP condition on goal position
def promp_condition(promp, T, xi, cov=None, t_c=1.0):
    """
    Compute conditioning on via points

    :param promp: Class object of ProMP
    :param T: Time sequence that parameterizes the trajectory
    :param xi: Via point coordinate
    :param cov: Covariance of the via point
    :param t_c: The time step of the via point
    :return: cpromp: Class object for ProMP after conditioning
    :return: mean: Mean trajectory after conditioning
    :return: std: An array of standard deviation after conditioning
    :return: cov: An array of covariance after conditioning
    """
    cpromp = promp.condition_position(xi, cov, t=t_c)

    mean = cpromp.mean_trajectory(T[0])
    cov = cpromp.cov_trajectory(T[0])
    std = np.sqrt(cpromp.var_trajectory(T[0]))

    return cpromp, mean, std, cov


# Move start to given position
def to_workspace_start(T, X, g_start):
    """
    Convert trajectory to start from a given workspace pose (only changes positions)

    :param T: Time sequence for the trajectory
    :param X: Data point
    :param g_start: SE(3) element for the starting pose
    :return X2: Positional trajectory after moving to the starting position
    """
    X2 = np.empty([X.shape[0], X.shape[1], 3])
    for i in range(X.shape[0]):

        if X.shape[1] == 2:
            for j in range(X.shape[1]):
                X2[i, j, :] = np.append(X[i, j, :], g_start[3, 3])
        else:
            X2[i, :, :] = X[i, :, :]

        X2[i, :, :] = X2[i, :, :] + (g_start[0:3, 3].T - X2[i, 0, :])

    return X2
