#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for ProMP in learning trajectory distribution from demonstration

@author: Sipu Ruan, 2022
"""

import primp.promp_interface as pl
from primp.util.data_parser import load_demos_position, load_demos
from primp.util.se3_util import get_exp_mapping, get_exp_coord
from primp.util.plot_util import draw_demos_position, plot_mean_std

import numpy as np
import matplotlib.pyplot as plt
import time
import os


def main():
    # ------ Parameters ------ #
    # Type of demonstration
    dataset_name = "panda_arm"
    demo_type = "simulation/letter_N"

    # Number of time step to be interpolated
    n_step = 50

    # Via pose deviation from initial, in exponential coordinates
    via_pose_deviation_exp_coord = np.concatenate([np.pi * 1e-2 * np.random.rand(3), 0.05 * np.random.rand(3)])

    # Scaling of via pose covariance
    cov_via_pose_scale = 1e-5
    # ------------------------ #

    data_path_prefix = os.getcwd() + "/../data/"
    load_prefix = data_path_prefix + dataset_name + "/" + demo_type

    # Load data and down sample
    g_demos = load_demos(load_prefix, n_step)
    T, X, w_rot = load_demos_position(n_step, load_prefix)

    # Add start/goal positions and convert to 3D workspace
    g_start = g_demos[1][0] @ get_exp_mapping(via_pose_deviation_exp_coord)
    g_via = g_demos[1][-1] @ get_exp_mapping(via_pose_deviation_exp_coord)

    w_fixed = w_rot[1, 0, :]
    xi_via = get_exp_coord(g_via, "PCG")
    xi_via[:3] = w_fixed

    # Learn distribution using ProMP
    print("ProMP learning trajectory distributions...")
    start = time.time()

    promp, mean, std, cov_joint = pl.promp_learn(T, X, 30)

    # Condition on via points
    print("Condition on via points")
    cpromp, mean_c, std_c, cov_joint_c = pl.promp_condition(promp, T, xi_via[3:], cov_via_pose_scale)

    elapsed_time = time.time() - start
    print("Ellapsed time: ", elapsed_time, " seconds")

    # Showing demonstration and learned distribution
    print("Plotting demonstration and learned distribution...")

    fig = plt.figure()
    axes = []
    axes.append(fig.add_subplot(1, 3, 1, projection="3d"))
    axes.append(fig.add_subplot(1, 3, 2, projection="3d"))
    axes.append(fig.add_subplot(1, 3, 3, projection="3d"))

    draw_demos_position(X, axes[0])
    plot_mean_std(mean, std, axes[1], "r--")
    plot_mean_std(mean_c, std_c, axes[2], "c--")

    axes[2].plot3D(g_start[0, 3], g_start[1, 3], g_start[2, 3], "bo")
    axes[2].plot3D(g_via[0, 3], g_via[1, 3], g_via[2, 3], "co")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plt.close("all")
    print("Start ProMP learning...")

    main()
