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

    # Load data and down sample
    T, X, w_rot = load_demos_position(50, load_prefix)

    # Load random via/goal poses for benchmark
    g_goal, cov_goal, t_via, g_via, cov_via = load_via_poses(result_path_prefix)
    n_trial = g_goal.shape[0]

    # Benchmark
    print("Benchmark: ProMP")
    print("Dataset: " + dataset_name)
    print("Demo type: " + demo_type)

    d_demo_promp = {"goal": np.zeros(n_trial), "via": np.zeros(n_trial)}
    d_via_promp = {"goal": np.zeros(n_trial), "via": np.zeros(n_trial)}
    for i in range(n_trial):
        print(str(i/n_trial*100.0) + "%")

        xi_goal = get_exp_coord(g_goal[i])
        xi_via = get_exp_coord(g_via[i])
        x_goal = xi_goal[3:]
        x_via = xi_via[3:]

        # Learn distribution using ProMP
        promp, mean, std, cov_joint = pl.promp_learn(T, X)

        # Condition on goal point
        promp_g, mean_g, std_g, cov_joint_g = pl.promp_condition(promp, T, x_goal, cov=cov_goal[i][3:, 3:], t_c=1.0)
        random_state = np.random.RandomState()
        x_samples_goal = promp_g.sample_trajectories(T[0], n_sample, random_state)

        # Condition on via point
        promp_v, mean_v, std_v, cov_joint_v = pl.promp_condition(promp_g, T, x_via, cov=cov_via[i][3:, 3:], t_c=t_via[i])
        random_state = np.random.RandomState()
        x_samples_via = promp_v.sample_trajectories(T[0], n_sample, random_state)

        # Compute distance to initial trajectory/demonstration
        d_demo_promp["goal"][i] = bu.evaluate_traj_distribution(x_samples_goal, X)
        d_demo_promp["via"][i] = bu.evaluate_traj_distribution(x_samples_via, X)

        # Compute distance to initial trajectory/demonstration
        d_via_promp["goal"][i] = bu.evaluate_desired_position(x_samples_goal, x_goal, 1.0)
        d_via_promp["via"][i] = bu.evaluate_desired_position(x_samples_via, x_via, t_via[i])

    # Store benchmark results
    dictionary = {
        "d_demo_promp_goal": d_demo_promp["goal"].tolist(),
        "d_demo_promp_via": d_demo_promp["via"].tolist(),
        "d_via_promp_goal": d_via_promp["goal"].tolist(),
        "d_via_promp_via": d_via_promp["via"].tolist()
    }
    write_json_file(dictionary, result_path_prefix + "/result_lfd_promp.json")

    # Print and record results
    file_path = result_path_prefix + "/result_lfd_promp.txt"
    with open(file_path, "w") as f:
        print("===========================", file=f)
        print(">>>> Condition on goal <<<<", file=f)
        print("---- Distance to demo (translation only):", file=f)
        print("ProMP (Tran)): " + str(np.mean(d_demo_promp["goal"])), file=f)
        print('---- Distance to desired pose (translation only):', file=f)
        print("ProMP (Tran)): " + str(np.mean(d_via_promp["goal"])), file=f)
        print('---------------------------------------------------------------', file=f)
        print('>>>> Condition on goal and a via pose <<<<', file=f)
        print("---- Distance to demo (translation only):", file=f)
        print("ProMP (Tran): " + str(np.mean(d_demo_promp["via"])), file=f)
        print('---- Distance to desired pose (translation only):', file=f)
        print("ProMP (Tran): " + str(np.mean(d_via_promp["via"])), file=f)


if __name__ == "__main__":
    dataset_name = "panda_arm"
    # dataset_name = "lasa_handwriting/pose_data"

    demo_types = bu.load_dataset_param(dataset_name)

    for demo_type in demo_types:
        benchmark(dataset_name, demo_type)
