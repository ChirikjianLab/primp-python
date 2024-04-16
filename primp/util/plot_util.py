import roboticstoolbox as rtb
from spatialmath import SE3, UnitQuaternion
import numpy as np
import matplotlib.pyplot as plt


# Plot reference frames
def plot_frame(g=np.identity(4), scale=1.0, ax=None):
    if ax is None:
        ax = plt.axes(projection="3d")

    base_t = scale * g[0:3, 0:3] + g[0:3, 3]

    ax.plot3D([g[0, 3], base_t[0, 0]], [g[1, 3], base_t[0, 1]], [g[2, 3], base_t[0, 2]], 'r', linewidth=2)
    ax.plot3D([g[0, 3], base_t[1, 0]], [g[1, 3], base_t[1, 1]], [g[2, 3], base_t[1, 2]], 'g', linewidth=2)
    ax.plot3D([g[0, 3], base_t[2, 0]], [g[1, 3], base_t[2, 1]], [g[2, 3], base_t[2, 2]], 'b', linewidth=2)


# Plot a trajectory
def plot_trajectory(g_traj, ax, line_style='k-', line_width=1):
    n_step = len(g_traj)

    # Plot trajectory
    for i in range(n_step-1):
        ax.plot3D([g_traj[i][0, 3], g_traj[i+1][0, 3]], [g_traj[i][1, 3], g_traj[i+1][1, 3]],
                  [g_traj[i][2, 3], g_traj[i+1][2, 3]], line_style, linewidth=line_width)


# Plot learned probability distributions with mean and standard deviation
def plot_mean_std(mean, std, ax, line_style):
    ax.plot3D(mean[:, 0], mean[:, 1], mean[:, 2], line_style)
    ax.plot(mean[:, 0] - std[:, 0], mean[:, 1] - std[:, 1], mean[:, 2] - std[:, 2], "g")
    ax.plot(mean[:, 0] + std[:, 0], mean[:, 1] + std[:, 1], mean[:, 2] + std[:, 2], "g")


# Demonstrate robot motion with given trajectory
def plot_robot_motion(g_traj=[np.identity(4)]):
    n_step = len(g_traj)
    robot = rtb.models.Panda()

    q_traj = np.zeros((n_step, 7))
    for i in range(n_step):
        quat = UnitQuaternion(g_traj[i][0:3, 0:3])
        se3_pose = SE3(g_traj[i][0, 3], g_traj[i][1, 3], g_traj[i][2, 3]) * quat.SE3()
        q_traj[i, :] = robot.ikine_LM(se3_pose).q

    robot.plot(q_traj)


# Draw demo trajectories
def draw_demos_position(X, ax):
    for pt in X:
        ax.plot3D(pt[:, 0].T, pt[:, 1].T, pt[:, 2].T)
