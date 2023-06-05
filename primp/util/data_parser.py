from primp.util.se3_util import homo2pose_quat, homo2pose_axang, get_exp_coord
from primp.util.interp_se3_trajectory import interp_se3_trajectory_svd
from primp.gora import GORA

import roboticstoolbox as rtb

import json
import csv
import numpy as np
import os


# Store learned distribution into file
def store_learned_density(g_mean, cov_joint, cov_step, file_name):
    num_steps = len(g_mean)
    mean_list = np.empty((num_steps, 4, 4))
    for i in range(len(g_mean)):
        mean_list[i] = g_mean[i]

    dictionary = {
        "num_steps": num_steps,
        "mean": mean_list.tolist(),
        "covariance_joint": cov_joint.tolist(),
        "covariance_step": cov_step.tolist()
    }

    write_json_file(dictionary, file_name + ".json")


def store_mean_csv(g_mean, file_name):
    mean_list = []
    for i in range(len(g_mean)):
        mean_list.append(homo2pose_axang(g_mean[i]))

    write_csv_file(mean_list, file_name + ".csv")


def store_start_goal(g_start, g_goal, file_name):
    dictionary = {
        "start_pose": homo2pose_quat(g_start).tolist(),
        "goal_pose": homo2pose_quat(g_goal).tolist()
    }

    write_json_file(dictionary, file_name + ".json")


def store_samples(g_samples, file_name):
    num_samples = len(g_samples)
    num_steps = len(g_samples[0])

    samples_quat = np.empty([num_samples, num_steps, 7])
    for i in range(num_samples):
        for j in range(num_steps):
            samples_quat[i][j] = homo2pose_quat(g_samples[i][j])

    dictionary = {
        "num_samples": num_samples,
        "num_steps": num_steps,
        "samples": samples_quat.tolist()
    }

    write_json_file(dictionary, file_name + ".json")


def load_demos(file_prefix, n_step, align_method="gora"):
    file_name = sorted(os.listdir(file_prefix))
    n_demo = len(file_name)

    # The list of demo trajectories
    g_demos = [[np.identity(4)] * n_step] * n_demo

    # Read demo trajectory one by one
    for i in range(n_demo):
        demo_data = load_json_file(file_prefix + "/" + file_name[i])

        g = np.array(demo_data["trajectory"])
        n_step_demo = demo_data["num_step"]

        if align_method == "gora":
            # Using GORA to align trajectories into unified timescale
            gora = GORA(g, n_step)
            gora.run()
            g_demos[i] = gora.get_optimal_trajectory()
        elif align_method == "interp":
            # Interpolation to unify the size of trajectory
            t0 = np.linspace(0, 1.0, n_step_demo)
            t_interp = np.linspace(0, 1.0, n_step)
            g_demos[i] = interp_se3_trajectory_svd(t0, g, t_interp)

    return g_demos


# Load generated via/goal poses
def load_via_poses(file_prefix):
    filename_prefix = "/trials_random"

    # Goal poses
    trials_data_goal = load_json_file(file_prefix + "/" + filename_prefix + "_goal.json")
    g_goal = np.array(trials_data_goal["g_goal"])
    cov_goal = np.array(trials_data_goal["cov_goal"])

    # Via poses
    trials_data_via = load_json_file(file_prefix + "/" + filename_prefix + "_via.json")
    t_via = np.array(trials_data_via["t_via"])
    g_via = np.array(trials_data_via["g_via"])
    cov_via = np.array(trials_data_via["cov_via"])

    return g_goal, cov_goal, t_via, g_via, cov_via


# Write data to .json file
def write_json_file(data, filename):
    with open(filename, "w") as outfile:
        json.dump(data, outfile)

    print("Write to file: " + filename)


# Load data from .json file
def load_json_file(filename):
    with open(filename, "r") as infile:
        data = json.load(infile)

    return data


# Write data to .csv file
def write_csv_file(data_array, filename):
    with open(filename, "w") as infile:
        writer = csv.writer(infile, delimiter=",")
        writer.writerows(data_array)

    print("Write to file: " + filename)


# Load data from .csv file
def load_csv_file(filename):
    data = []
    with open(filename, "r") as infile:
        reader = csv.reader(infile, delimiter=",")
        for row in reader:
            data.append(row)

    return data


# Load Panda robot with random start/goal configurations and interpolated trajectory in between
def load_robot(n_step=50):
    robot = rtb.models.Panda()

    # Random start/goal configurations
    q_start = robot.qr
    q_goal = robot.qz
    q_traj = rtb.jtraj(q_start, q_goal, n_step)

    # Store end effector pose trajectory as initial mean
    mean_init = [np.identity(4)] * n_step
    for i in range(n_step):
        mean_init[i] = robot.fkine(q_traj.q[i, :]).A

    return robot, q_traj, mean_init


# Compute index of via pose
def convert_time_to_index(t, n_step):
    idx = int(np.floor(t * n_step))
    if idx <= 0:
        idx = 0
    elif idx >= n_step - 1:
        idx = n_step - 1

    return idx


# Load demonstrated trajectories
def load_demos_position(num_step, file_prefix):
    g_demos = load_demos(file_prefix, num_step)

    num_demos = len(g_demos)
    X = np.empty([num_demos, num_step, 3])
    w_rot = np.empty([num_demos, num_step, 3])
    t = np.linspace(0, 1.0, X.shape[1])
    T = np.tile(t, (X.shape[0], 1))

    # Read demo trajectory one by one
    for i in range(num_demos):
        num_demo_step = len(g_demos[i])
        x = np.empty([num_demo_step, 3])
        w = np.empty([num_demo_step, 3])

        for j in range(num_demo_step):
            xi = get_exp_coord(np.array(g_demos[i][j]), "PCG")

            X[i, j, :] = xi[3:]
            w_rot[i, j, :] = xi[:3]

    return T, X, w_rot


# Store learned distribution into file
def store_promp(mean, cov_step, cov_joint, file_name):
    dictionary = {
        "num_steps": mean.shape[0],
        "mean": mean.tolist(),
        "covariance_step": cov_step.tolist(),
        "covariance_joint": cov_joint.tolist(),
    }

    write_json_file(dictionary, file_name + ".json")


def store_promp_mean_csv(mean, file_name):
    write_csv_file(mean, file_name + ".csv")