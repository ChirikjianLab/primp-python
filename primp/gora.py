from primp.util.interp_se3_trajectory import interp_se3_trajectory_svd
from primp.util.finite_difference import df_vect
from primp.util.se3_util import norm_se3
import numpy as np
from scipy.integrate import cumtrapz


class GORA:
    """
    Globally Optimal Reparameterization Algorithm (GORA) for SE(3) trajectory data
    1. Recovering a finite difference approximation of the change between adjacent frames of an SE(3) trajectory
    2. globally minimizing the functional of the form ``J = \int_0^1 g(\tau) \dot{\tau} dt''
    3. reparameterizing the sequence with the optimal time steps

    @authors: Sipu Ruan, Thomas Mitchel
    """

    def __init__(self, g_init, n_step=50):
        self._g_init = g_init

        self._n_step_init = len(g_init)
        self._t0 = np.linspace(0, 1.0, self._n_step_init)

        self._n_step = n_step
        self._tau_opt = np.linspace(0, 1.0, self._n_step)
        self._g_opt = [np.identity(4)] * self._n_step

    def run(self):
        """
        Run the main routines
        """
        g = self.__g_tau()
        self.__temporal_reparam(g)
        self._g_opt = interp_se3_trajectory_svd(self._t0, self._g_init, self._tau_opt)

    def get_optimal_trajectory(self):
        """
        Retrieve the optimal trajectory
        """
        return self._g_opt

    def get_optimal_time(self):
        """
        Retrieve the optimal temporal reparameterization
        """
        return self._tau_opt

    def get_cost_functional(self, tau, g_tau):
        """
        Compute cost functional
        :param tau: Parameterization of the trajectory
        :param g_tau: SE(3) trajectory parameterized by tau
        :return: Cost functional value
        """
        d_tau = np.gradient(tau, self._t0)
        cost = np.trapz(d_tau ** 2.0 * g_tau, self._t0)
        return cost

    def __g_tau(self):
        """
        Generate g(\tau) for the input of Euler-Lagrange equation
        :return: Computed g(tau)
        """
        dg = df_vect(self._t0, self._g_init, 1, 2, 0)
        g = np.zeros(self._n_step_init)

        for i in range(self._n_step_init):
            g[i] += norm_se3(np.linalg.inv(self._g_init[i]) @ dg[i]) ** 2.0

        return np.asarray(g)

    def __temporal_reparam(self, g):
        """
        Compute the globally optimal time sequence by Theorem 1
        :param g: Parameterized g(tau)
        """
        f = cumtrapz(g ** 0.5, self._t0, initial=0)
        f /= f[-1]
        self._tau_opt = np.interp(self._tau_opt, f, self._t0)
