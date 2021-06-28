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
        Second derivative of the kernel w.r.t. r
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


# class LinearKernel(Kernel):
#     """
#     Calculates covariance as: (x'y+c)^d
#     """

#     def __init__(self, d, c=1):
#         super().__init__(d=d, c=c)
#         self._name = 'Linear kernel'

#     def __str__(self):
#         mess = self._name + ':\n'
#         mess += super().__str__()
#         return mess

#     def _dist(self, xa, xb):
#         c = self.hyperparameters.get('c')
#         self.P = np.einsum('ia,ib', xa, xb) + c
#         return self.P

#     def K(self, xa, xb):

#         p = self._dist(xa, xb)
#         d = self.hyperparameters.get('d')

#         return np.power(p, d)

#     def condition_gradient(self, x, obs):
#         N, M = x.shape
#         p = self._dist(x, x)
#         d = self.hyperparameters.get('d')

#         Kab = d * (d - 1) * p**(d - 2)
#         Kaa = d * np.diag(p)**(d - 1)

#         L = np.linalg.cholesky(Kab)
#         H_ = np.eye(M) + L.T.dot(np.diag(np.sum(x * x, axis=0) / Kaa).dot(L))
#         zeta = L.dot(np.linalg.solve(H_, L.T.dot(
#             (np.sum(x * df, axis=0) / Kaa).reshape(-1, 1))))
#         self.alpha = df / Kaa - x * zeta.T / Kaa

#         return self.alpha

#     def dK_inv(self, x):

#         N, M = x.shape
#         p = self._dist(x, x)
#         d = self.hyperparameters.get('d')

#         k_ab = d * (d - 1) * p**(d - 2)
#         k_aa = d * np.diag(p)**(d - 1)

#         L = np.linalg.cholesky(k_ab)
# #         k_aa_=1/k_aa

#         x_ = x / k_aa
#         k_aa = np.diag(1 / k_aa)

#         X_ = np.zeros((N * M, M))
#         for a in range(M):
#             X_[a * N:(a + 1) * N, a] = x_[:, a]

#         X_L = X_.dot(L)


# #         M=k_ab+x.T.dot(x_)
#         M = np.eye(M) + L.T.dot(x.T.dot(x_).dot(L))

#         G_ = X_L.dot(np.linalg.solve(M, X_L.T)) + np.kron(k_aa, np.eye(N))
#         return G_

# #         X_=x_.reshape(-1,1)

#     def dK(self, xa, xb):
#         Na, Ma = xa.shape
#         Nb, Mb = xb.shape
#         d = self.hyperparameters.get('d')

#         p = self._dist(xa, xb)

#         k_ab = d * (d - 1) * p**(d - 2)
#         k_aa = np.diag(d * np.diag(p)**(d - 1))


# #         G=np.zeros((Na*Ma,Nb*Mb))
# #         for a in range(M):
# #             for b in range(M):
# #                 G[a*N:(a+1)*N,b*N:(b+1)*N]+=k_ab[a,b]*np.outer(xa[:,a],xb[:,b])

# #                 if a==b:
# #                     G[a*N:(a+1)*N,b*N:(b+1)*N]+=k_aa[a,a]*np.eye(N)

#         Xa = np.zeros((Na * Ma, Ma))
#         for a in range(Ma):
#             Xa[a * Na:(a + 1) * Na, a] = xa[:, a]

#         Xb = np.zeros((Nb * Mb, Mb))
#         for b in range(Mb):
#             Xb[b * Nb:(b + 1) * Nb, b] = xb[:, b]

#         G = Xa.dot(k_ab.dot(Xb.T)) + np.kron(k_aa, np.eye(N))

#         return G
