#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark for ProMP in learning trajectory distribution from demonstration

@author: Sipu Ruan
"""

import numpy as np
import dtw


def load_dataset_param(dataset_name):
    """
    Load the parameters for datasets

    :param dataset_name: The name of the dataset: "panda_arm", "lasa_handwriting/pose_data"
    :return: The type list of the demonstration
    """
    demo_type = dict()

    # Type of demonstration
    if dataset_name == "panda_arm":
        demo_type = {'simulation/circle', 'simulation/letter_N', 'simulation/letter_U', 'simulation/letter_S',
                     'real/pouring/default', 'real/scooping/default', 'real/transporting/default',
                     'real/opening/sliding', 'real/opening/rotating_left', 'real/opening/rotating_down'}
    elif dataset_name == "lasa_handwriting/pose_data":
        demo_type = {'Angle', 'BendedLine', 'CShape', 'DoubleBendedLine', 'GShape', 'heee', 'JShape', 'JShape_2',
                     'Khamesh', 'Leaf_1', 'Leaf_2', 'Line', 'LShape', 'NShape', 'PShape', 'RShape', 'Saeghe', 'Sharpc',
                     'Sine', 'Snake', 'Spoon', 'Sshape', 'Trapezoid', 'Worm', 'WShape', 'Zshape'}

    return demo_type


def evaluate_traj_distribution(traj_res, traj_init):
    """
    Evaluate the generated trajectory distribution

    :param traj_res: The resulting (generated) trajectory
    :param traj_init: The initial trajectory
    :return: The distance between the generated and initial trajectory mean
    """
    n_demo = traj_init.shape[0]
    n_sample = traj_res.shape[0]
    n_step = traj_init.shape[1]

    # Distance to initial trajectory mean (translation only)
    d_mean_traj = 0.0
    for k in range(n_demo):
        for j in range(n_sample):
            tran_init = traj_init[k, :, :]
            tran_res = traj_res[j, :, :]

            d_mean_traj += dtw.dtw(tran_res, tran_init).distance

    # Take average over the whole trajectory
    d_mean_traj /= (n_demo * n_sample * n_step)

    return d_mean_traj


def evaluate_desired_position(traj_res, x_desired, t_via):
    """
    Evaluate similarity between result and desired pose.

    :param traj_res: Resulting (Generated) trajectory
    :param x_desired: The desired position
    :param t_via: The time step of via point
    :return: The distance from the desired position
    """
    n_sample = traj_res.shape[0]
    n_step = traj_res.shape[1]

    # Compute index in trajectory closest to desired time step
    step_via = int(np.floor(t_via * n_step))
    if step_via < 0:
        step_via = 0
    elif step_via >= n_step-1:
        step_via = n_step-1

    # Compute distance to desired pose
    d_desired_pose = 0.0
    for j in range(n_sample):
        tran_res = traj_res[j, step_via, :]
        d_desired_pose += np.linalg.norm(x_desired - tran_res)

    # Take average over all samples
    d_desired_pose /= n_sample

    return d_desired_pose
