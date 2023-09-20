#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark for ProMP in learning trajectory distribution from demonstration
@author: Sipu Ruan, 2022
"""

import primp.promp_interface as pl
from primp.util.data_parser import load_demos_position, load_via_poses, write_json_file
from primp.util.se3_util import get_exp_coord
from primp.util import benchmark_util as bu

import numpy as np
import os


def benchmark(dataset_name, demo_type):
    n_sample = 50

    data_path_prefix = os.getcwd() + "/../data/"
    result_path_prefix = os.getcwd() + "/../result/benchmark/" + dataset_name + "/" + demo_type
    load_prefix = data_path_prefix + dataset_name + "/" + demo_type + "/"

    # Parameter for ProMP
    n_weight = [10, 20, 30, 40, 50]
    n_param = len(n_weight)

    # Load data and down sample
    T, X, w_rot = load_demos_position(50, load_prefix)

    # Load random via/goal poses for benchmark
    g_goal, cov_goal, t_via, g_via, cov_via = load_via_poses(result_path_prefix)
    n_trial = g_goal.shape[0]

    # Benchmark
    print("Benchmark: ProMP")
    print("Dataset: " + dataset_name)
    print("Demo type: " + demo_type)

    d_demo = {"goal": np.zeros((n_param, n_trial)), "via": np.zeros((n_param, n_trial))}
    d_via = {"goal": np.zeros((n_param, n_trial)), "via": np.zeros((n_param, n_trial))}
    for j in range(n_param):
        print("Number of weights per dimension: " + str(n_weight[j]))

        for i in range(n_trial):
            print(str(i/n_trial*100.0) + "%")

            x_goal = g_goal[i][:3,3]
            x_via = g_via[i][:3,3]

            # Learn distribution using ProMP
            promp, mean, std, cov_joint = pl.promp_learn(T, X, n_weight[j])

            # Condition on goal point
            promp_g, mean_g, std_g, cov_joint_g = pl.promp_condition(promp, T, x_goal, cov=cov_goal[i][3:, 3:], t_c=1.0)
            random_state = np.random.RandomState()
            x_samples_goal = promp_g.sample_trajectories(T[0], n_sample, random_state)

            # Condition on via point
            promp_v, mean_v, std_v, cov_joint_v = pl.promp_condition(promp_g, T, x_via, cov=cov_via[i][3:, 3:], t_c=t_via[i])
            random_state = np.random.RandomState()
            x_samples_via = promp_v.sample_trajectories(T[0], n_sample, random_state)

            # Compute distance to initial trajectory/demonstration
            d_demo["goal"][j][i] = bu.evaluate_traj_distribution(x_samples_goal, X)
            d_demo["via"][j][i] = bu.evaluate_traj_distribution(x_samples_via, X)

            # Compute distance to initial trajectory/demonstration
            d_via["goal"][j][i] = bu.evaluate_desired_position(x_samples_goal, x_goal, 1.0)
            d_via["via"][j][i] = bu.evaluate_desired_position(x_samples_via, x_via, t_via[i])

    # Store benchmark results
    dictionary = {
        "format": "n_param x n_trial",
        "num of weights": n_weight,
        "d_demo_goal": d_demo["goal"].tolist(),
        "d_demo_via": d_demo["via"].tolist(),
        "d_via_goal": d_via["goal"].tolist(),
        "d_via_via": d_via["via"].tolist()
    }
    write_json_file(dictionary, result_path_prefix + "/result_lfd_promp.json")

    # Print and record results
    file_path = result_path_prefix + "/result_lfd_promp.txt"
    with open(file_path, "w") as f:
        print("===========================", file=f)
        print(">>>> Condition on goal <<<<", file=f)
        print("---- Distance to demo (translation only):", file=f)
        print("ProMP (Tran): " + str(np.mean(d_demo["goal"], axis=1)), file=f)
        print('---- Distance to desired pose (translation only):', file=f)
        print("ProMP (Tran): " + str(np.mean(d_via["goal"], axis=1)), file=f)
        print('---------------------------------------------------------------', file=f)
        print('>>>> Condition on goal and a via pose <<<<', file=f)
        print("---- Distance to demo (translation only):", file=f)
        print("ProMP (Tran): " + str(np.mean(d_demo["via"], axis=1)), file=f)
        print('---- Distance to desired pose (translation only):', file=f)
        print("ProMP (Tran): " + str(np.mean(d_via["via"], axis=1)), file=f)


if __name__ == "__main__":
    # dataset_name = "panda_arm"
    dataset_name = "lasa_handwriting/pose_data"

    demo_types = bu.load_dataset_param(dataset_name)

    for demo_type in demo_types:
        benchmark(dataset_name, demo_type)
