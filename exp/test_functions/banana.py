# Banana test function for arbitrary dimension

import numpy as np
from scipy.stats import special_ortho_group
import warnings


class Banana:
    def __init__(self, dim: int, a: np.ndarray, seed: bool = None):
        """
        Creates a random rotation of the following function in D dimensions:
            f(x) = exp (- 0.5*(x_1^2 + (a_0 x_1^2 + a_1 x_2 + a_2)^2 + sum_{i=3}^D a_i x_i^2 ))
        :param a: parameters of the banana
        """
        if a.size < dim + 1:
            warnings.warn('parameter vector is too short, remaining ones will be filled with 1.')
            a = np.concatenate([a.flatten(), np.ones((dim+1 - a.size))])

        self.dim = dim
        self.a = a

        # sample a random rotation matrix
        if seed is not None:
            self.R = special_ortho_group(dim, seed).rvs()
        else:
            self.R = np.eye(dim)

    def _x_trafo(self, x):
        x = self.R @ x
        if x.ndim == 1:
            x = x[:, None]
        x[1, :] = self.a[0] * x[0, :] ** 2 + self.a[1] * x[1, :] + self.a[2]
        x[2:, :] = self.a[3:, None] * x[2:, :]
        return x

    def f(self, x):
        "Some unnormalized density"
        return np.exp(-self.pot_energy(x)).squeeze()

    def pot_energy(self, x):
        "The negative log of the above, that is, the potential energy"
        arg = self._x_trafo(x)
        return 0.5 * (arg ** 2).sum(axis=0).squeeze()

    def grad_e(self, x):
        grad = self._x_trafo(x)
        grad[0, :] = grad[0, :] + grad[1, :] * 2 * x[0, :] * self.a[0]
        grad[1, :] = grad[1, :] * self.a[1]
        return grad
