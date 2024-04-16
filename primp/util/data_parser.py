from primp.util.se3_util import homo2pose_quat, homo2pose_axang, get_exp_coord
from primp.util.interp_se3_trajectory import interp_se3_trajectory_svd
from primp.gora import GORA

import roboticstoolbox as rtb

import json
import csv
import numpy as np
import os


def store_learned_density(g_mean, cov_joint, cov_step, file_name):
    """
    Store the learned trajectory distribution (density) into file. Format: .json
    :param g_mean: The trajectory mean
    :param cov_joint: The covariance matrix of joint distribution
    :param cov_step: An array of covariance
    :param file_name: The name of file for storage
    """
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
    """
    Store the mean trajectory into file. Format: .csv
    :param g_mean: The mean trajectory as an array of poses
    :param file_name: The name of file for storage
    """
    mean_list = []
    for i in range(len(g_mean)):
        mean_list.append(homo2pose_axang(g_mean[i]))

    write_csv_file(mean_list, file_name + ".csv")


def store_start_goal(g_start, g_goal, file_name):
    """
    Store start and goal poses into file. Format: .json
    :param g_start: The starting pose
    :param g_goal: The goal pose
    :param file_name: The name of file for storage
    """
    dictionary = {
        "start_pose": homo2pose_quat(g_start).tolist(),
        "goal_pose": homo2pose_quat(g_goal).tolist()
    }

    write_json_file(dictionary, file_name + ".json")


def store_samples(g_samples, file_name):
    """
    Store the sampled trajectories from the distribution into file. Format: .json
    :param g_samples: The sampled trajectories as an array of trajectories
    :param file_name: The name of file for storage
    """
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
    """
    Load demonstrations from file
    :param file_prefix: The prefix of the file name
    :param n_step: The number of time steps
    :param align_method: The method of temporal alignment: "gora", "interp"
    :return g_demos: The array of converted demonstration trajectories
    """
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
    """
    Load via-point poses from file (only two via points)
    :param file_prefix: The prefix of file name
    :return t_via_1: Time step for via point 1
    :return g_via_1: Pose of via point 1
    :return cov_via_1: Covariance of via point 1
    :return t_via_2: Time step for via point 2
    :return g_via_2: Pose of via point 2
    :return cov_via_2: Covariance of via point 2
    """
    filename_prefix = "/trials_random"

    # Via-point poses
    trials_data = load_json_file(file_prefix + "/" + filename_prefix + "_via_1.json")
    t_via_1 = np.array(trials_data["t_via"])
    g_via_1 = np.array(trials_data["g_via"])
    cov_via_1 = np.array(trials_data["cov_via"])

    trials_data = load_json_file(file_prefix + "/" + filename_prefix + "_via_2.json")
    t_via_2 = np.array(trials_data["t_via"])
    g_via_2 = np.array(trials_data["g_via"])
    cov_via_2 = np.array(trials_data["cov_via"])

    return t_via_1, g_via_1, cov_via_1, t_via_2, g_via_2, cov_via_2


def write_json_file(data, filename):
    """
    Write data into .json file
    :param data: Data to be written
    :param filename: The file name for storage
    """
    with open(filename, "w") as outfile:
        json.dump(data, outfile)

    print("Write to file: " + filename)


def load_json_file(filename):
    """
    Load data from .json file
    :param filename: The file name for data
    :return data: The loaded data
    """
    with open(filename, "r") as infile:
        data = json.load(infile)

    return data


def write_csv_file(data_array, filename):
    """
    Write data into .csv file
    :param data_array: Data to be written
    :param filename: The file name for storage
    """
    with open(filename, "w") as infile:
        writer = csv.writer(infile, delimiter=",")
        writer.writerows(data_array)

    print("Write to file: " + filename)


def load_csv_file(filename):
    """
    Load data from .csv file
    :param filename: The file name for data
    :return data: The loaded data
    """
    data = []
    with open(filename, "r") as infile:
        reader = csv.reader(infile, delimiter=",")
        for row in reader:
            data.append(row)

    return data


def load_robot(n_step=50):
    """
    Load Panda robot with random start/goal configurations and interpolated trajectory in between
    :param n_step: Number of time steps
    :return robot: Panda model object
    :return q_traj: Joint-space trajectory
    :return mean_init: The initial mean trajectory of the end effector
    """
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


def convert_time_to_index(t, n_step):
    """
    Convert time step into index
    :param t: The time step
    :param n_step: The number of time steps
    :return idx: The index in the time sequence
    """
    idx = int(np.floor(t * n_step))
    if idx <= 0:
        idx = 0
    elif idx >= n_step - 1:
        idx = n_step - 1

    return idx


def load_demos_position(num_step, file_prefix):
    """
    Load demonstrated trajectories (Only position)
    :param num_step: Number of time steps
    :param file_prefix: The prefix of file name
    :return T: Time sequence of trajectory
    :return X: Data points in the trajectory
    :return w_rot: Exponential coordinate for rotation
    """
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


def store_promp(mean, cov_step, cov_joint, file_name):
    """
    Store learned distribution into file. Format: .json
    :param mean: Mean trajectory
    :param cov_step: Covariance at each time step
    :param cov_joint: The covariance of joint distribution
    :param file_name: The file name for storage
    """
    dictionary = {
        "num_steps": mean.shape[0],
        "mean": mean.tolist(),
        "covariance_step": cov_step.tolist(),
        "covariance_joint": cov_joint.tolist(),
    }

    write_json_file(dictionary, file_name + ".json")


def store_promp_mean_csv(mean, file_name):
    """
    Store mean trajectory learned by ProMP into file. Format: .csv
    :param mean: Mean trajectory
    :param file_name: The file name for storage
    """
    write_csv_file(mean, file_name + ".csv")