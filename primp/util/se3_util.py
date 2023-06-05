import numpy as np
from scipy.spatial.transform import Rotation as R
from math import sin, cos, acos
import warnings


def homo2pose_quat(g=np.identity(4)):
    """
    Convert SE(3) homogeneous matrix to pose vector, rotation is represented by unit Quaternion
    :param g: SE(3) element
    :return: Pose vector, quaternion order: [x, y, z, w]
    """
    rot = R.from_matrix(g[0:3, 0:3])

    # Quaternion: (x, y, z, w)
    quat = rot.as_quat()

    pose = np.concatenate((g[0:3, 3].T, quat), axis=None)

    return pose


def homo2pose_axang(g=np.identity(4)):
    """
    Convert SE(3) homogeneous matrix to pose vector, rotation is represented by axis-angle
    :param g: SE(3) element
    :return: Pose vector, axis-angle order: [axis-x, axis-y, axis-z, angle]
    """
    rot = R.from_matrix(g[0:3, 0:3])

    # Axis-angle: (x, y, z, theta)
    rotvec = rot.as_rotvec()
    axang = np.concatenate((rotvec, np.linalg.norm(rotvec)), axis=None)

    pose = np.concatenate((g[0:3, 3].T, axang), axis=None)

    return pose


def get_exp_mapping(exp_coord=np.zeros(6), group_name="SE"):
    """
    Get the exponential mapping from Lie algebra (exponential coordinates) to Lie group
    :param exp_coord: Exponential coordinates of the form [\omega, v]
    :param group_name: "SE" (default), "PCG"
    :return: Homogeneous transformation matrix

    See also: expm_se3, expm_so3
    """
    g = np.identity(4)

    if group_name == "SE":
        g = expm_se3(exp_coord)

    elif group_name == "PCG":
        g[0:3, 0:3] = expm_so3(exp_coord[0:3])
        g[0:3, 3] = exp_coord[3:].T

    else:
        warnings.WarningMessage("Group not supported.")

    return g


def get_exp_coord(g=np.identity(4), group_name="SE"):
    """
    Get exponential coordinates from Lie group to Lie algebra
    :param g: Homogeneous transformation matrix for poses
    :param group_name: "SE" (default), "PCG"
    :return: Exponential coordinates of the form [\omega, v]

    See also: logm_se3, logm_so3
    """
    exp_coord = np.zeros(6)

    if group_name == "SE":
        xi_hat = logm_se3(g)
        exp_coord = np.block([vex(xi_hat[0:3, 0:3]), xi_hat[0:3, 3].T])

    elif group_name == "PCG":
        w_hat = logm_so3(g[0:3, 0:3])
        exp_coord = np.block([vex(w_hat), g[0:3, 3].T])

    else:
        warnings.WarningMessage("Group not supported.")

    return exp_coord


def expm_se3(xi=np.zeros(6)):
    """
    Closed-form exponential from se(3) to SE(3)
    :param xi: Element in se(3) of the form [\omega, v]
    :return: Homogeneous transformation matrix for SE(3)
    """
    g = np.identity(4)
    g[0:3, 0:3] = expm_so3(xi[0:3])
    g[0:3, 3] = jacobian_so3_l(xi[0:3]) @ xi[3:].T

    return g


def expm_so3(w=np.zeros(3)):
    """
    Closed-form exponential for so(3) into SO(3)
    :param w: Vector for rotation
    :return: Rotation matrix
    """
    rot = np.identity(3)
    if np.linalg.norm(w) > 1e-15:
        rot = np.identity(3) + sin(np.linalg.norm(w)) / np.linalg.norm(w) * skew(w) + \
              (1 - cos(np.linalg.norm(w))) / (np.linalg.norm(w) ** 2.0) * (skew(w) @ skew(w))

    return rot


def logm_se3(g=np.identity(4)):
    """
    Closed-form logarithm from SE(3) to se(3)
    :param g: Homogeneous transformation matrix for SE(3)
    :return: Element in se(3) in the matrix form
    """
    rot = g[0:3, 0:3]
    t = g[0:3, 3]

    w_hat = logm_so3(rot)
    w = vex(w_hat)

    xi_hat = np.zeros((4, 4))
    xi_hat[0:3, 0:3] = w_hat
    xi_hat[0:3, 3] = jacobian_inv_so3_l(w) @ t

    return xi_hat


def logm_so3(rot=np.identity(3)):
    """
    Closed-form logarithm from SO(3) to so(3)
    :param rot: Rotation matrix as element in SO(3)
    :return: Element in so(3) in the matrix form
    """
    val = (np.trace(rot) - 1.0) / 2.0

    # Force cosine value within range
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0

    theta = acos(val)

    w_hat = np.zeros((3, 3))
    if sin(theta) > 1e-15:
        w_hat = 1 / 2 * theta / sin(theta) * (rot - rot.T)

    return w_hat


def jacobian_so3_l(w=np.zeros(3)):
    """
    Closed-form left Jacobian of SO(3)
    :param w: Vector for the rotation
    :return: Closed-form left Jacobian of SO(3)
    """
    jac_l = np.identity(3)
    if np.linalg.norm(w) > 1e-15:
        jac_l = np.identity(3) + (1.0 - cos(np.linalg.norm(w))) / (np.linalg.norm(w) ** 2.0) * skew(w) +\
                (np.linalg.norm(w) - sin(np.linalg.norm(w))) / (np.linalg.norm(w) ** 3) * (skew(w) @ skew(w))

    return jac_l


def jacobian_inv_so3_l(w=np.zeros(3)):
    """
    Closed-form left inverse Jacobian of SO(3)
    :param w: Vector for the rotation
    :return: Closed-form left inverse Jacobian of SO(3)
    """
    jac_inv_l = np.identity(3)
    if np.linalg.norm(w) > 1e-15:
        jac_inv_l = np.identity(3) - 1 / 2 * skew(w) + \
                    (1 / (np.linalg.norm(w) ** 2.0) -
                     (1 + cos(np.linalg.norm(w))) / (2 * np.linalg.norm(w) * sin(np.linalg.norm(w)))) * \
                    (skew(w) @ skew(w))

    return jac_inv_l


def adjoint_group(g=np.identity(4), group_name="SE"):
    """
    Adjoint operator for groups SE(3) or PCG(3)
    :param g: Group element in homogeneous matrix form
    :param group_name: "SE" (default), "PCG"
    :return: Adjoint operator
    """
    ad = np.zeros((6, 6))

    # Block for rotation part
    rot = g[0:3, 0:3]
    ad[0:3, 0:3] = rot

    # Other blocks
    if group_name == "SE":
        t_matrix = np.array([[0, -g[2, 3], g[1, 3]], [g[2, 3], 0, -g[0, 3]], [-g[1, 3], g[0, 3], 0]])
        ad[3:, 0:3] = t_matrix @ rot
        ad[3:, 3:] = rot

    elif group_name == "PCG":
        ad[3:, 3:] = np.identity(3)

    else:
        warnings.WarningMessage("Group not supported.")

    return ad


def skew(w=np.zeros(3)):
    """
    Construct skew-symmetric matrix from vector
    :param w: Vector of size 3
    :return: Skew-symmetric matrix
    """
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def vex(w_hat=np.zeros((3, 3))):
    """
    Extract vector form from skew-symmetric matrix
    :param w_hat: Skew-symmetric matrix
    :return: Vector form
    """
    return np.array([-w_hat[1, 2], w_hat[0, 2], -w_hat[0, 1]])


def norm_se3(g):
    """
    Weighted norm of SE(3)
    :param g: an SE(3) element
    :return: Weighted norm
    """
    # Inertia matrix for unit sphere of unit mass
    I = (2 / 5) * np.identity(3)
    J = (1 / 2) * np.trace(I) * np.identity(3) - I

    # Unit mass
    M = 1

    # Weight
    w = np.block([[J, np.zeros((3, 1))], [np.zeros((1, 3)), M]])

    # Weighted norm
    weighted_norm = np.sqrt(np.trace(g.T @ w @ g))

    return weighted_norm
