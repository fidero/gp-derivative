import numpy as np
from .base_kernel import StationaryKernel
from typing import Union


class RBF(StationaryKernel):
    """
    Calculates covariance as: exp(-(x-y).T W (x-y)/2)
    """

    def __init__(self, input_dim: int, w: Union[float, np.ndarray]):
        """
        :param input_dim: Input dimension
        :param w: float or np.ndarray
        """

        super().__init__(input_dim, w)

    def K(self, xa, xb):
        W = self.hyperparameters["w"]
        # self.XTWX = xa.T.dot(W*xb)
        XTWX = xa.T.dot(W * xb)
        self.XTWX = XTWX
        D = (
            np.sum(xa * W * xa, axis=0, keepdims=True).T
            + np.sum(xb * W * xb, axis=0, keepdims=True)
            - 2 * XTWX
        )
        return np.exp(-0.5 * D)

    def _dK_dr(self, xa, xb):
        """
        Derivative w.r.t. r
        """
        return -self.K(xa, xb) / 2.0
        # return -self.K(xa,xb)

    def _d2K_dr2(self, xa, xb):
        """
        Second derivative of the kernel w.r.t. r
        """
        # return self.K(xa,xb)
        return self.K(xa, xb) / 4.0

    def _d3K_dr3(self, xa, xb):
        """
        Third derivative of the kernel w.r.t. r
        """
        # return self.K(xa,xb)
        return -self.K(xa, xb) / 8.0


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

    kern = RBF(D, w=w)

    # print(issubclass(type(kern),StationaryKernel),type(kern).__name__)

    G = kern._dKd_explicit(X, X)
    # G2= kern._dKd_explicit(X[:,:1],X)
    # L=np.linalg.cholesky(G+1e-6*np.eye(D*M))
    print(np.linalg.eigvalsh(G))

    plt.figure()
    plt.imshow(G)

    plt.show()
