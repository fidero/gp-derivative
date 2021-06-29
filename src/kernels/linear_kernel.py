import numpy as np
from .base_kernel import InnerProductKernel
from typing import Union


class LinearKernel(InnerProductKernel):
    """
    Calculates covariance as: ((x-c).T W (y-c))^p
    """

    def __init__(
        self, input_dim: int, w: Union[float, np.ndarray], c: float = 0.0, p: int = 2
    ):
        """
        :param input_dim: Input dimension
        :param w: float or np.ndarray
        :param p: polynomial factor
        :param c: np.ndarray reference point
        """

        super().__init__(input_dim, w, c, p=p)

    def K(self, xa, xb):
        c = self.hyperparameters["c"]
        W = self.hyperparameters["w"]
        p = self.hyperparameters["p"]

        self.XTWX = (xa - c).T.dot(W * (xb - c))
        return np.power(self.XTWX, p)


    def _dK_dr(self, xa, xb):
        """
        Derivative w.r.t. r
        """
        c = self.hyperparameters["c"]
        W = self.hyperparameters["w"]
        p = self.hyperparameters["p"]

        self.XTWX = (xa - c).T.dot(W * (xb - c))
        return p * np.power(self.XTWX, p - 1)

    def _d2K_dr2(self, xa, xb):
        """
        Second derivative of the kernel w.r.t. r
        """
        c = self.hyperparameters["c"]
        W = self.hyperparameters["w"]
        p = self.hyperparameters["p"]

        self.XTWX = (xa - c).T.dot(W * (xb - c))
        return p * (p - 1) * np.power(self.XTWX, p - 2)

    def _d3K_dr3(self, xa, xb):
        """
        Third derivative of the kernel w.r.t. r
        """
        c = self.hyperparameters["c"]
        W = self.hyperparameters["w"]
        p = self.hyperparameters["p"]

        if p>2:
            self.XTWX = (xa - c).T.dot(W * (xb - c))
            return p * (p - 1) * (p - 2) * np.power(self.XTWX, p - 3)
        else:
            return np.zeros((xa.shape[1],xb.shape[1]))


class FastLinearKernel(LinearKernel):
    """
    Calculates covariance as: ((x-c).T W (y-c))^p
    Assume observations are valid for faster inference, as is the case for linear algebra.
    """

    def __init__(
        self, input_dim: int, w: Union[float, np.ndarray], p:int = 2,c: float = 0.0):
        """
        :param input_dim: Input dimension
        :param w: float or np.ndarray
        :param p: polynomial factor
        :param c: np.ndarray reference point
        """
        if p not in {1,2}:
            raise ValueError(f"p must be either 1 or 2, got: {p}")

        super().__init__(input_dim, w, c, p=p)



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    D = 4
    M = 8
    X = 2 + np.random.randn(D, M)  # /np.sqrt(D)
    # X=X[:,1:]-X[:,:-1]
    X_ = X.reshape(-1, 1, order="f")
    w = 1 / np.pi

    kern = LinearKernel(D, w=w, p=2)

    G = kern._dKd_explicit(X, X)
    # G2= kern._dKd_explicit(X[:,:1],X)
    # L=np.linalg.cholesky(G+1e-6*np.eye(D*M))
    print(np.linalg.eigvalsh(G))

    plt.figure()
    plt.imshow(G)

    plt.show()

