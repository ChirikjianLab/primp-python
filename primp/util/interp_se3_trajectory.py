import numpy as np
from primp.util.finite_difference import df_vect

from scipy.linalg import svd


def interp_se3_trajectory_svd(t0, x_traj, t):
    """
    Compute interpolation along an SE(3) trajectory using SVD

    Reference: "An SVD-Based Projection Method for Interpolation on SE(3). Calin Belta, Vijay Kumar. 2002."

    :param t0: Original time sequence
    :param x_traj: Original SE(3) trajectory
    :param t: Desired time sequence
    :return: Interpolated SE(3) trajectory

    @author: Thomas Mitchel, 2018
    @maintainer: Sipu Ruan, 2022
    """

    num_t0 = len(t0)
    num_traj = int(x_traj.shape[2] / 4)

    # Pre-allocate array
    x_interp = np.zeros((len(t), 4, 4 * num_traj))

    # Pre-allocate weights
    I = (2 / 5) * np.identity(3)
    J = (1 / 2) * np.trace(I) * np.identity(3) - I

    # Pre-compute derivative
    dx_traj = df_vect(t0, x_traj, 1, 2, 0)

    # Trajectory-wise interpolation in SE(3)
    count = -1
    for i in range(1, num_t0):
        if i == num_t0 - 1:
            int_v = t[(t0[i - 1] <= t) & (t <= t0[i])]
        else:
            int_v = t[(t0[i - 1] <= t) & (t < t0[i])]

        if int_v.shape[0] != 0:
            dt = t0[i] - t0[i - 1]
            for j in range(0, num_traj):
                g0 = x_traj[i - 1, :, 4 * j: 4 * j + 4]
                g1 = x_traj[i, :, 4 * j:4 * j + 4]
                dg0 = dx_traj[i - 1, :, 4 * j: 4 * j + 4]
                dg1 = dx_traj[i, :, 4 * j: 4 * j + 4]

                # Interp pose (position and orientation)
                dx = g1 - g0
                dv = dg1 - dg0

                M3 = 6 * ((dg0 + dg1) / (dt ** 2)) - 12 * (dx / (dt ** 3))
                M2 = (dv / dt) - M3 * (t0[i] + t0[i - 1]) / 2
                M1 = dg0 - M3 * ((t0[i - 1] ** 2) / 2) - M2 * t0[i - 1]
                M0 = g0 - M3 * ((t0[i - 1] ** 3) / 6) - M2 * ((t0[i - 1] ** 2) / 2) - M1 * t0[i - 1]

                for l in range(int_v.shape[0]):
                    g_itp = np.zeros((4, 4))
                    g_itp[3, 3] = 1
                    t_eval = int_v[l]
                    M = M0 + M1 * t_eval + (M2 / 2) * (t_eval ** 2) + (M3 / 6) * (t_eval ** 3)
                    U, S, V = svd(np.matmul(M[0:3, 0:3], J))
                    g_itp[0:3, 0:3] = np.matmul(U, V)
                    g_itp[0:3, 3] = M[0:3, 3]
                    x_interp[count + l + 1, :, (4 * j):((4 * j) + 4)] = g_itp
            count += len(int_v)

    return x_interp