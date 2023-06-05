import primp.primp as primp
from primp.util.se3_util import get_exp_mapping
from primp.util.data_parser import load_robot, convert_time_to_index
from primp.util.plot_util import plot_frame, plot_trajectory

import numpy as np
import matplotlib.pyplot as plt


def run_demo(group_name="SE"):
    # ------ Parameters ------ #
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

    # Initial covariance
    cov_init = [1e-5 * np.diag([1, 1, 1, 4, 4, 4])] * n_step
    # ------------------------ #

    # Initiate robot and show its movement
    print("Loading robot with random configurations...")
    robot, q_traj, mean_init = load_robot(n_step)

    # Define goal and via poses with uncertainty
    step_via = convert_time_to_index(t_via, n_step)
    g_goal = mean_init[-1] @ get_exp_mapping(via_pose_deviation_exp_coord)
    cov_goal = cov_via_pose_scale * np.diag([4, 4, 4, 1, 1, 1])
    g_via = mean_init[step_via] @ get_exp_mapping(via_pose_deviation_exp_coord)
    cov_via = cov_via_pose_scale * np.diag([4, 4, 4, 1, 1, 1])

    # ------ Main routines ------ #
    print("Encode joint distribution")
    density = primp.PrIMP(mean_init=mean_init, cov_init=cov_init, group_name=group_name)

    print("Condition on a via pose")
    density.condition_via_pose(g_goal, cov_goal, 1.0)
    density.condition_via_pose(g_via, cov_via, t_via)

    mean_cond, cov_cond = density.get_joint_pdf()
    g_samples_cond = density.get_samples(n_sample)

    # ------ Plot trajectories ------ #
    # -- Plot mean trajectory and random samples
    plt.figure()
    ax = plt.axes(projection="3d")

    plot_trajectory(mean_init, ax, 'b', 1.5)
    plot_trajectory(mean_cond, ax, 'm', 1.5)
    for g_sample in g_samples_cond:
        plot_trajectory(g_sample, ax, 'm', 1)

    plot_frame(g_goal, 0.1, ax)
    plot_frame(g_via, 0.1, ax)

    ax.set_aspect("equal")


if __name__ == "__main__":
    print("Start PRIMP learning...")

    # Group: PCG(3)
    print("PCG(3) formulation")
    run_demo("PCG")

    plt.show()
