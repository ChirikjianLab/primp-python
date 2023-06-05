#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for ProMP in LASA handwriting trajectory learning

@author: ruan
"""
from movement_primitives.data import load_lasa
from movement_primitives.promp import ProMP

import numpy as np
import matplotlib.pyplot as plt
import json


def promp_learn(T, X, idx, axes, shape_name):
    h = int(idx / width)
    w = int(idx % width) * 2
    axes[h, w].set_title(shape_name)
    axes[h, w].plot(X[:, :, 0].T, X[:, :, 1].T)

    promp = ProMP(n_weights_per_dim=30, n_dims=X.shape[2])
    promp.imitate(T, X)
    mean = promp.mean_trajectory(T[0])
    cov = promp.cov_trajectory(T[0])
    std = np.sqrt(promp.var_trajectory(T[0]))

    axes[h, w + 1].plot(mean[:, 0], mean[:, 1], c="r")
    axes[h, w + 1].plot(mean[:, 0] - std[:, 0], mean[:, 1] - std[:, 1], c="g")
    axes[h, w + 1].plot(mean[:, 0] + std[:, 0], mean[:, 1] + std[:, 1], c="g")

    # axes[h, w + 1].set_xlim(axes[h, w].get_xlim())
    # axes[h, w + 1].set_ylim(axes[h, w].get_ylim())
    axes[h, w].get_yaxis().set_visible(False)
    axes[h, w].get_xaxis().set_visible(False)
    axes[h, w + 1].get_yaxis().set_visible(False)
    axes[h, w + 1].get_xaxis().set_visible(False)

    return promp, mean, std, cov


# Down sample to limit the size of trajectory
def down_sample(T, X, rate):
    d_sample = T.shape[1] / (int(rate * T.shape[1]) - 1)
    idx_sample = range(0, T.shape[1], int(d_sample))

    T_d = T[:, 0 : T.shape[1] : int(d_sample)]
    X_d = X[:, 0 : T.shape[1] : int(d_sample), :]

    return T_d, X_d


# Move start to given position
def to_workspace_start(T, X, g_start):
    X2 = np.empty([X.shape[0], X.shape[1], 3])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X2[i, j, :] = np.append(X[i, j, :], g_start[3, 3])

        X2[i, :, :] = X2[i, :, :] + (g_start[0:3, 3].T - X2[i, 0, :])

    return X2


# ProMP condition on goal position
def promp_condition(promp, T, g, t_c, idx, axes, shape_name):
    cpromp = promp.condition_position(g_start[0:3, 3].T, t=t_c)

    mean = cpromp.mean_trajectory(T[0])
    cov = cpromp.cov_trajectory(T[0])
    std = np.sqrt(cpromp.var_trajectory(T[0]))

    h = int(idx / width)
    w = int(idx % width) * 2
    axes[h, w + 1].plot(mean[:, 0], mean[:, 1], c="r")
    axes[h, w + 1].plot(mean[:, 0] - std[:, 0], mean[:, 1] - std[:, 1], c="g")
    axes[h, w + 1].plot(mean[:, 0] + std[:, 0], mean[:, 1] + std[:, 1], c="g")

    axes[h, w + 1].plot(g[0, 3], g[1, 3], "bo")

    return mean, std, cov


# Store learned distribution into file
def store_promp(mean, cov_step, cov_joint, file_name):
    dictionary = {
        "mean": mean.tolist(),
        "covariance_step": cov_step.tolist(),
        "covariance_joint": cov_joint.tolist(),
    }

    with open(file_name + ".json", "w") as outfile:
        json.dump(dictionary, outfile)


if __name__ == "__main__":
    plt.close("all")

    width = 2
    height = 5

    fig1, axes1 = plt.subplots(int(height), int(width * 2))

    for i in range(width * height):
        print(i)

        T, X, Xd, Xdd, dt, shape_name = load_lasa(i)
        T_d, X_d = down_sample(T, X, 0.06)

        # Add start/goal positions and convert to 3D workspace
        g_start = np.array([[1, 0, 0, 46], [0, 1, 0, -23], [0, 0, 1, 30], [0, 0, 0, 1]])
        g_goal = np.array([[1, 0, 0, 36], [0, 1, 0, 23], [0, 0, 1, 30], [0, 0, 0, 1]])

        X_w = to_workspace_start(T_d, X_d, g_start)

        # Learn distribution using ProMP
        promp, mean, std, cov_joint = promp_learn(T_d, X_w, i, axes1, shape_name)

        # Condition on goal position
        mean_c, std_c, cov_joint_c = promp_condition(
            promp, T_d, g_start, 0.0, i, axes1, shape_name
        )

        # Store mean and covariance to file
        cov_step_c = np.empty([std_c.shape[0], mean_c.shape[1], mean_c.shape[1]])
        for j in range(std_c.shape[0]):
            cov_step_c[j] = np.diag(std_c[j, :] ** 2)
        file_name = "trajectory_promp_lasa_" + shape_name
        store_promp(mean_c, cov_step_c, cov_joint_c, file_name)

    plt.tight_layout()
    plt.show()
