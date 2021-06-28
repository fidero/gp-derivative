import numpy as np
from .base_kernel import InnerProductKernel
from typing import Union


class ExponentialKernel(InnerProductKernel):
    """
    Calculates covariance as: exp((x-c).T W (y-c))
    """

    def __init__(self, input_dim: int, w: Union[float, np.ndarray], c: float = 0.0):
        """
        :param input_dim: Input dimension
        :param w: float or np.ndarray
        :param p: polynomial factor
        :param c: np.ndarray reference point
        """

        super().__init__(input_dim, w, c)

    def K(self, xa, xb):
        c = self.hyperparameters["c"]
        W = self.hyperparameters["w"]

        self.XTWX = (xa - c).T.dot(W * (xb - c))
        return np.exp(self.XTWX)

    def _dK_dr(self, xa, xb):
        """
        Derivative w.r.t. r
        """
        c = self.hyperparameters["c"]
        W = self.hyperparameters["w"]

        self.XTWX = (xa - c).T.dot(W * (xb - c))
        return np.exp(self.XTWX)

    def _d2K_dr2(self, xa, xb):
        """
        Second derivative of the kernel w.r.t. r
        """
        c = self.hyperparameters["c"]
        W = self.hyperparameters["w"]

        self.XTWX = (xa - c).T.dot(W * (xb - c))
        return np.exp(self.XTWX)

    def _d3K_dr3(self, xa, xb):
        """
        Second derivative of the kernel w.r.t. r
        """
        c = self.hyperparameters["c"]
        W = self.hyperparameters["w"]

        self.XTWX = (xa - c).T.dot(W * (xb - c))
        return np.exp(self.XTWX)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    D = 4
    M = 8
    X = 2 + np.random.randn(D, M) / np.sqrt(D)
    # X=X[:,1:]-X[:,:-1]
    X_ = X.reshape(-1, 1, order="f")
    w = 1 / np.pi

    kern = ExponentialKernel(D, w=w, p=2)

    G = kern._dKd_explicit(X, X)
    # G2= kern._dKd_explicit(X[:,:1],X)
    # L=np.linalg.cholesky(G+1e-6*np.eye(D*M))
    print(np.linalg.eigvalsh(G))

    plt.figure()
    plt.imshow(G)

    plt.show()
