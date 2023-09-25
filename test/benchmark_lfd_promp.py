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
    t_via_1, g_via_1, cov_via_1, t_via_2, g_via_2, cov_via_2 = load_via_poses(result_path_prefix)
    n_trial = g_via_1.shape[0]

    # Benchmark
    print("Benchmark: ProMP")
    print("Dataset: " + dataset_name)
    print("Demo type: " + demo_type)

    d_demo = {"via_1": np.zeros((n_param, n_trial)), "via_2": np.zeros((n_param, n_trial))}
    d_via = {"via_1": np.zeros((n_param, n_trial)), "via_2": np.zeros((n_param, n_trial))}
    for j in range(n_param):
        print("Number of weights per dimension: " + str(n_weight[j]))

        for i in range(n_trial):
            print(str(i/n_trial*100.0) + "%")

            x_via_1 = g_via_1[i][:3, 3]
            x_via_2 = g_via_2[i][:3, 3]

            # Learn distribution using ProMP
            promp, mean, std, cov_joint = pl.promp_learn(T, X, n_weight[j])

            # Condition on via-point poses
            promp_g, mean_g, std_g, cov_joint_g = pl.promp_condition(promp, T, x_via_1, cov=cov_via_1[i][3:, 3:],
                                                                     t_c=t_via_1[i])
            random_state = np.random.RandomState()
            x_samples_1 = promp_g.sample_trajectories(T[0], n_sample, random_state)

            promp_v, mean_v, std_v, cov_joint_v = pl.promp_condition(promp_g, T, x_via_2, cov=cov_via_2[i][3:, 3:],
                                                                     t_c=t_via_2[i])
            random_state = np.random.RandomState()
            x_samples_2 = promp_v.sample_trajectories(T[0], n_sample, random_state)

            # Compute distance to initial trajectory/demonstration
            d_demo["via_1"][j][i] = bu.evaluate_traj_distribution(x_samples_1, X)
            d_demo["via_2"][j][i] = bu.evaluate_traj_distribution(x_samples_2, X)

            # Compute distance to initial trajectory/demonstration
            d_via["via_1"][j][i] = bu.evaluate_desired_position(x_samples_1, x_via_1, t_via_1[i])
            d_via["via_2"][j][i] = bu.evaluate_desired_position(x_samples_2, x_via_2, t_via_2[i])

    # Store benchmark results
    dictionary = {
        "format": "n_param x n_trial",
        "num_of_weights": n_weight,
        "d_demo_via_1": d_demo["via_1"].tolist(),
        "d_demo_via_2": d_demo["via_2"].tolist(),
        "d_via_via_1": d_via["via_1"].tolist(),
        "d_via_via_2": d_via["via_2"].tolist()
    }
    write_json_file(dictionary, result_path_prefix + "/result_lfd_promp.json")

    # Print and record results
    file_path = result_path_prefix + "/result_lfd_promp.txt"
    with open(file_path, "w") as f:
        print("===========================", file=f)
        print(">>>> Condition on 1 via point <<<<", file=f)
        print("---- Distance to demo (translation only):", file=f)
        print("ProMP (Tran): " + str(np.mean(d_demo["via_1"], axis=1)), file=f)
        print('---- Distance to desired pose (translation only):', file=f)
        print("ProMP (Tran): " + str(np.mean(d_via["via_1"], axis=1)), file=f)
        print('---------------------------------------------------------------', file=f)
        print('>>>> Condition on 2 via points <<<<', file=f)
        print("---- Distance to demo (translation only):", file=f)
        print("ProMP (Tran): " + str(np.mean(d_demo["via_2"], axis=1)), file=f)
        print('---- Distance to desired pose (translation only):', file=f)
        print("ProMP (Tran): " + str(np.mean(d_via["via_2"], axis=1)), file=f)


if __name__ == "__main__":
    # Name of dataset
    dataset_names = ["panda_arm", "lasa_handwriting/pose_data"]

    for dataset_name in dataset_names:
        # Name of demo types
        demo_types = bu.load_dataset_param(dataset_name)

        for demo_type in demo_types:
            # Run benchmark for each demo type for each dataset
            benchmark(dataset_name, demo_type)
