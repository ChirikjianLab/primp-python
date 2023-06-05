import primp.primp as primp
from primp.util.data_parser import load_demos, convert_time_to_index
from primp.util.plot_util import plot_frame, plot_trajectory
from primp.util.se3_util import get_exp_mapping

import numpy as np
import matplotlib.pyplot as plt
import os


def run_demo(group_name="SE"):
    # ------ Parameters ------ #
    # Type of demonstration
    dataset_name = "panda_arm"
    demo_type = "simulation/letter_N"

    # Number of time step to be interpolated
    n_step = 50

    # Number of sampled trajectories from joint distribution
    n_sample = 5

    # Define via pose with uncertainty
    t_via = 0.36

    # Via pose deviation from initial, in exponential coordinates
    via_pose_deviation_exp_coord = np.concatenate([np.pi * 1e-2 * np.random.rand(3), 0.1 * np.random.rand(3)])

    # Scaling of via pose covariance
    cov_via_pose_scale = 1e-5
    # ------------------------ #

    # Paths for data
    data_path_prefix = os.getcwd() + "/../data/"
    load_prefix = data_path_prefix + dataset_name + "/" + demo_type

    # Load demos
    g_demos = load_demos(load_prefix, n_step, "gora")

    # Define desired via poses
    step_via = convert_time_to_index(t_via, n_step)
    g_goal = g_demos[0][-1] @ get_exp_mapping(via_pose_deviation_exp_coord)
    cov_goal = cov_via_pose_scale * np.random.rand() * np.diag([4, 4, 4, 1, 1, 1])
    g_via = g_demos[0][step_via] @ get_exp_mapping(via_pose_deviation_exp_coord)
    cov_via = cov_via_pose_scale * np.random.rand() * np.diag([4, 4, 4, 1, 1, 1])

    print("Demonstrated trajectories loaded!")

    # ------ Main routines ------ #
    print("Encoding demos into joint distribution...")

    density = primp.PrIMP(g_demos, group_name=group_name)
    mean_joint, cov_joint = density.get_joint_pdf()

    # ------ Adaptation to a goal pose ------#
    print("Adapt to new goal...")

    density = primp.PrIMP(g_demos, group_name=group_name)

    # Condition on goal pose
    density.condition_via_pose(g_goal, cov_goal, 1.0)
    mean_cond_goal, cov_cond_goal = density.get_joint_pdf()
    g_samples_cond_goal = density.get_samples(n_sample)

    # ------ Adaptation to multiple via poses ------ #
    print("Adapt to multiple new via poses...")

    density = primp.PrIMP(g_demos, group_name=group_name)

    # Condition on goal and poses
    density.condition_via_pose(g_goal, cov_goal, 1.0)
    density.condition_via_pose(g_via, cov_via, t_via)
    mean_cond_via, cov_cond_via = density.get_joint_pdf()
    g_samples_cond_via = density.get_samples(n_sample)

    # ------ Plots ------#
    # -- Plot demos and mean trajectory with frames
    plt.figure()
    ax = plt.axes(projection="3d")

    for g_demo in g_demos:
        plot_trajectory(g_demo, ax, line_style='k-', line_width=1)

    plot_trajectory(mean_joint, ax, line_style='b', line_width=1.5)

    for i in range(n_step):
        plot_frame(mean_joint[i], 0.02, ax)

    ax.set_aspect("equal")

    # -- Plot conditioned probability on goal pose
    fig = plt.figure()
    fig.suptitle("Group: " + group_name + "(3); Condition on goal pose")
    ax = plt.axes(projection="3d")

    plot_trajectory(mean_joint, ax, line_style='b', line_width=1.5)
    plot_trajectory(mean_cond_goal, ax, line_style='m', line_width=1.5)

    # Samples from conditional probability
    for g_sample in g_samples_cond_goal:
        plot_trajectory(g_sample, ax, line_style='m', line_width=1)

    plot_frame(g_goal, 0.1, ax)

    ax.set_aspect("equal")

    # -- Plot conditioned probability on goal and via pose
    fig = plt.figure()
    fig.suptitle("Group: " + group_name + "(3); Condition on goal and a via pose")
    ax = plt.axes(projection="3d")

    plot_trajectory(mean_joint, ax, line_style='b', line_width=1.5)
    plot_trajectory(mean_cond_via, ax, line_style='m', line_width=1.5)

    # Samples from conditional probability
    for g_sample in g_samples_cond_via:
        plot_trajectory(g_sample, ax, line_style='m', line_width=1)

    plot_frame(g_goal, 0.1, ax)
    plot_frame(g_via, 0.1, ax)

    ax.set_aspect("equal")


if __name__ == "__main__":
    print("Start PRIMP learning...")

    # Group: PCG(3)
    print("PCG(3) formulation")
    run_demo("PCG")

    plt.show()
