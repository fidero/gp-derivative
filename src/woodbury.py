import numpy as np


def _build_C(d1, d2, d3):
    """Builds a symmetric matrix C with provided diagonals: C=[[d1,d2],[d2,d3]]"""
    C = np.diag(np.hstack((d1, d3))) + np.kron(np.fliplr(np.eye(2)), np.diag(d2))
    return C


class SymmetricWoodbury:
    """Symmetric diagonal + low rank matrix for fast inversion, developed for Hessian inference.
    :param b: np.ndarray[D,1]
    :param U: np.ndarray[D,N]
    :param c1: np.ndarray[D,1]
    :param c2: np.ndarray[D,1]
    """

    def __init__(self, b, U, c1, c2):
        self.b = b
        self.U = U
        self.c1 = np.atleast_2d(c1)
        self.c2 = np.atleast_2d(c2)

    def __matmul__(self, v):
        return self.b * v + self.U @ (self.C @ (self.U.T @ v))

    def __mul__(self, v):
        return self.__matmul__(v)

    # def __rtruediv__(self, v):
    #     # Must subclass numpy ndarray...
    #     return self.invmul(v)

    def matmul(self, v):
        return self.b * v + self.U @ (self.C @ (self.U.T @ v))

    def invmul(self, v):
        V = self.U / self.b
        T = self.Cinv + V.T.dot(self.U)
        return v / self.b - V @ np.linalg.solve(T, V.T.dot(v))


    @property
    def full(self):
        return np.diag(self.b[:,0]) + self.U.dot(self.C.dot(self.U.T))
    

    # def __get__(self):
    #     return np.diag(self.b[:,0]) + self.U.dot(self.C.dot(self.U.T))

    @property
    def C(self):
        return _build_C(self.c1[0], self.c2[0], np.zeros_like(self.c1)[0])

    @property
    def Cinv(self):
        return _build_C(
            np.zeros_like(self.c1)[0],
            1 / self.c2[0],
            -(self.c1 / self.c2 ** 2)[0],
        )


if __name__ == "__main__":
    ################
    # Small test of identities
    ################

    import matplotlib.pyplot as plt

    D = 5
    N = 2

    b = 3 * np.ones((D, 1))
    c1 = 2 + np.abs(np.random.randn(N))
    c2 = 1 + np.abs(np.random.randn(N))


    U = np.random.randn(D, 2 * N)

    W = SymmetricWoodbury(b, U, c1, c2)

    v = np.random.randn(D, 1)
    print(type(v))
    W @ v


    print((W @ v).T)
    print((W * v).T)

    Wv= W @ v

    print(v.T)
    print((W.invmul(Wv).T))
 