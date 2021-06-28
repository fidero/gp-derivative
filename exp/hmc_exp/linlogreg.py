import numpy as np


class Logistic:
    """
    Logistic regression model.
    adapted from https://github.com/modestbayes/hamiltonian/blob/master/models.py
    """

    def __init__(self, data_X, data_y, alpha):
        """
        data_X: (N, D); data_y: (N, 1), alpha: float
        """
        self.X = data_X
        self.y = data_y
        self.alpha = alpha

    def _loglikelihood(self, beta):
        """
        Log logistic regression likelihood of beta.
        beta: (D, M)
        """
        return -(np.log(np.exp(self.X @ beta) + 1.0)).sum(axis=0) + np.dot(
            np.transpose(self.y), np.dot(self.X, beta)
        )

    def _grad_loglikelihood(self, beta):
        """
        Gradient of Log logistic regression likelihood wrt beta.
        beta: (D, M)
        """
        return self.X.T @ (self.y - 1.0 / (1.0 + np.exp(-self.X @ beta)))

    def _logprior(self, beta):
        """
        Log independent Gaussian prior density of beta.
        """
        return -0.5 / self.alpha * (beta ** 2).sum(axis=0)

    def _grad_logprior(self, beta):
        """
        Gradient of Log independent Gaussian prior density wrt beta.
        """
        return -beta / self.alpha

    def get_dim(self):
        return self.X.shape[1]

    def energy(self, beta):
        return -(self._loglikelihood(beta) + self._logprior(beta))

    def gradient(self, beta):
        return -(self._grad_loglikelihood(beta) + self._grad_logprior(beta))

    def scale(self):
        """
        Typical scale of the energy is N / D (causing it to get really large with increasing N)
        """
        return self.X.shape[1] / self.X.shape[0]


def generate_logreg_data(n, beta, prior, sigma=2.0):
    """Generate logistic regression data.

    # Arguments
        n: number of observations
        beta: regression coefficient vector
        prior: Uniform or Gaussian
        sigma: parameter of prior (width of hypercube or Gaussian std. dev.)

    # Returns
        A list of design matrix and response vector
    """
    d = beta.shape[0]
    if prior == "Uniform":
        X = np.random.rand(n, d) * sigma - 1.0 / sigma
    elif prior == "Gaussian":
        X = np.sqrt(sigma) * np.random.randn(n, d)

    mu = 1.0 / (1.0 + np.exp(-np.dot(X, beta)))
    Y = np.random.binomial(1, mu) * 1.0
    return X, Y
