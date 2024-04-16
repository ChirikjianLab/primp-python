from primp.util.se3_util import get_exp_mapping, get_exp_coord, adjoint_group
from primp.util.se3_distribution import get_mean, get_covariance
from primp.gora import GORA
import warnings
import numpy as np


class PrIMP:
    """
    PRIMP class for PRobabilistically-Informed Motion Primitives. It encodes demonstrated trajectories in Lie groups,
    adapts to novel via-point poses.

    @author: Sipu Ruan
    """
    def __init__(self, g_demos=None, mean_init=None, cov_init=None, group_name="SE"):
        # Initial input values
        self._g_demos = g_demos
        self._mean_init = mean_init
        self._cov_init = cov_init
        self._n_demo = 0
        self._group_name = group_name

        # Number of time steps
        self._n_step = 0
        if self._g_demos is not None:
            self._n_step = len(self._g_demos[0])
            self._n_demo = len(self._g_demos)
            self._mean_init = [np.identity(4)] * self._n_step
            self._cov_init = [np.zeros((6, 6))] * self._n_step

            # Learn initial distribution from demo
            self.__learn()

        elif self._mean_init is not None:
            self._n_step = len(self._mean_init)

        else:
            warnings.WarningMessage("List of either demonstration trajectories or "
                                    "initial distribution should be given.")

        # Joint distribution
        self._mean_joint = self._mean_init
        self._cov_joint = np.zeros((6 * self._n_step, 6 * self._n_step))
        self._cov_joint_inv = np.zeros((6 * self._n_step, 6 * self._n_step))

        self._t_idx_matrix = np.zeros((6, 6 * self._n_step))

        # Compute joint distribution from the initial input
        self.__compute_joint_pdf()

    def get_joint_pdf(self):
        """
        Retrieve joint distribution

        :return: self._mean_joint: Mean value of the joint distribution
        :return: self._cov_joint: Covariance matrix of the joint distribution
        """
        return self._mean_joint, self._cov_joint

    def get_covariance_step(self):
        """
        Retrieve covariance of each step

        :return: cov_step: Covariance matrix of each step
        """
        cov_step = np.empty((self._n_step, 6, 6))
        for i in range(self._n_step):
            cov_step[i] = self._cov_joint[i*6:(i+1)*6, i*6:(i+1)*6]

        return cov_step

    def get_samples(self, n_sample=5):
        """
        Generate random samples from Gaussian based on joint distribution

        :param: n_sample: Number of samples
        :return: Array of SE(3) trajectory samples
        """
        # Generate sample variables from joint distribution
        x_var = np.zeros(6*self._n_step)
        x_sample = np.random.multivariate_normal(x_var, self._cov_joint, n_sample)

        # Compute SE(3) trajectory samples
        g_samples = [[np.identity(4)] * self._n_step] * n_sample
        for i in range(n_sample):
            g_sample = [np.identity(4)] * self._n_step
            for j in range(self._n_step):
                g_sample[j] = self._mean_joint[j] @ get_exp_mapping(x_sample[i, 6*j:6*(j+1)], self._group_name)
            g_samples[i] = g_sample

        return g_samples

    def condition_via_pose(self, g_via=np.identity(4), cov_via=np.identity(6), t_via=1.0):
        """
        Compute conditional distribution based on via pose

        :param g_via: Mean of the via pose
        :param cov_via: Covariance of the via pose
        :param t_via: Time parameter of the desired pose
        """
        self.__compute_time_step_idx(t_via)

        k_gain = self._cov_joint @ self._t_idx_matrix.T @ \
                 np.linalg.inv(self._t_idx_matrix @ self._cov_joint @ self._t_idx_matrix.T + cov_via)

        # Mean variable after conditioning: condition on the desired pose g_via
        x_mu = k_gain @ get_exp_coord(np.linalg.inv(self._mean_joint[self._t_idx]) @ g_via, self._group_name)

        # Update covariance after conditioning
        self._cov_joint = self._cov_joint - k_gain @ self._t_idx_matrix @ self._cov_joint

        # Compute trajectory mean after conditioning
        for i in range(self._n_step):
            self._mean_joint[i] = self._mean_joint[i] @ get_exp_mapping(x_mu[6*i:6*(i+1)], self._group_name)

    def __compute_joint_pdf(self):
        """
        Compute covariance of the joint distribution from initial mean and covariance
        """
        for i in range(self._n_step):
            idx_start = i*6
            idx_end = (i+1)*6

            # Fill in the blocks of the joint covariance
            if i != self._n_step-1:
                # Pose difference between adjacent steps
                d_mu_i = np.linalg.inv(self._mean_init[i]) @ self._mean_init[i+1]
                adjoint_mat = adjoint_group(d_mu_i)
                cov_tilde = adjoint_mat @ self._cov_init[i+1] @ adjoint_mat.T

                self._cov_joint_inv[idx_start:idx_end, idx_start:idx_end] = np.linalg.inv(self._cov_init[i]) + \
                                                                            np.linalg.inv(cov_tilde)
                self._cov_joint_inv[idx_start:idx_end, idx_start+6:idx_end+6] = -np.linalg.inv(
                    self._cov_init[i+1] @ adjoint_mat.T)
                self._cov_joint_inv[idx_start+6:idx_end+6, idx_start:idx_end] = -np.linalg.inv(
                    adjoint_mat @ self._cov_init[i+1])
            else:
                self._cov_joint_inv[idx_start:idx_end, idx_start:idx_end] = np.linalg.inv(self._cov_init[i])

        # Ensure it is symmetric
        self._cov_joint_inv = (self._cov_joint_inv + self._cov_joint_inv.T) / 2.0
        self._cov_joint = np.linalg.inv(self._cov_joint_inv)

    def __compute_time_step_idx(self, t_via=1.0):
        """
        Compute index of the closest step in the trajectory with the desired pose

        :param t_via: time parameter for the desired via pose, [0,1]
        """
        # Locate the closest step with the desired via pose
        self._t_idx = int(np.floor(t_via * self._n_step))
        if self._t_idx < 0:
            self._t_idx = 0
        elif self._t_idx >= self._n_step-1:
            self._t_idx = self._n_step-1

        # Compute time step index in matrix form
        p_matrix = np.zeros(self._n_step)
        p_matrix[self._t_idx] = 1.0
        self._t_idx_matrix = np.kron(p_matrix, np.identity(6))

    def __learn(self):
        """
        Learn trajectory distribution in Lie groups from demonstrations
        """
        if self._g_demos is None:
            return

        # Extract absolute and relative frames for each demo, group each step into a set of SE(3)
        g_steps = [[np.identity(4)] * self._n_demo] * self._n_step
        g_steps_rel = [[np.identity(4)] * self._n_demo] * (self._n_step-1)

        for i in range(self._n_step):
            g_step = [np.identity(4)] * self._n_demo
            g_step_rel = [np.identity(4)] * self._n_demo

            # Get the set of absolute and relative poses for each step
            for j in range(self._n_demo):
                g_step[j] = self._g_demos[j][i]
                if i < self._n_step-1:
                    g_step_rel[j] = np.linalg.inv(self._g_demos[j][i]) @ self._g_demos[j][i+1]

            g_steps[i] = g_step
            if i < self._n_step - 1:
                g_steps_rel[i] = g_step_rel

        # Compute mean using original trajectory
        g_mean = []
        for i in range(self._n_step):
            g, flag = get_mean(g_steps[i], self._group_name)

            if flag:
                g_mean.append(g)

        # Apply GORA for optimal mean trajectory
        gora = GORA(np.array(g_mean), self._n_step)
        gora.run()
        g_mean = gora.get_optimal_trajectory()

        # Convert format for mean_init
        for i in range(self._n_step):
            self._mean_init[i] = g_mean[i]

        # Compute covariance using relative frames
        self._cov_init[0] = get_covariance(g_steps[0], self._mean_init[0], self._group_name)
        for i in range(self._n_step-1):
            if i != self._n_step-1:
                self._cov_init[i+1] = get_covariance(g_steps_rel[i], group_name=self._group_name)

            # Add regularization to avoid singularity
            self._cov_init[i] += 1e-5 * np.identity(6)
