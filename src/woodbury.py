import numpy as np


def _build_C(d1, d2, d3):
    """Builds a symmetric matrix C with provided diagonals: C=[[d1,d2],[d2,d3]]"""
    # N=c1.shape[0]
    # C=np.zeros((2*N,2*N))
    # C[:N,:N] = np.diag(d1)
    # C[:N,N:] = np.diag(d2)
    # C[N:,:N] = np.diag(d2)
    # C[N:,N:] = np.diag(d3)
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

    def __rtruediv__(self, v):
        # Must subclass numpy ndarray...
        return self.invmul(v)

    def matmul(self, v):
        return self.b * v + self.U @ (self.C @ (self.U.T @ v))

    def invmul(self, v):
        V = self.U / self.b
        T = self.Cinv + V.T.dot(self.U)
        return v / self.b - V @ np.linalg.solve(T, V.T.dot(v))

    # def __repr__(self):
    #     return np.diag(self.b[:,0]) + self.U.dot(self.C.dot(self.U.T))

    @property
    def full(self):
        # print(self.U.shape,self.C.shape)
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
            # np.zeros(self.c1.shape[0]),
            1 / self.c2[0],
            -(self.c1 / self.c2 ** 2)[0],
        )


if __name__ == "__main__":

    # print(np.ndarray.__dir__)
    # print(dir(np.ndarray))
    import matplotlib.pyplot as plt

    D = 5
    N = 2

    b = 3 * np.ones((D, 1))
    c1 = 2 + np.abs(np.random.randn(N))
    c2 = 1 + np.abs(np.random.randn(N))
    # np.ones((N,1))

    U = np.random.randn(D, 2 * N)

    W = SymmetricWoodbury(b, U, c1, c2)

    v = np.random.randn(D, 1)
    print(type(v))
    W @ v

    # print(v.T)
    # print((W @ W.invmul(v)).T)
    # print(W.invmul( W @ v).T)

    print((W @ v).T)
    print((W * v).T)

    Wv= W @ v

    print(v.T)
    print((W.invmul(Wv).T))
    
    # print((Wv/W).T)
    # print((W/Wv).T)
    # print(U)
    # print(W.full)
    # C = np.zeros((2*N,2*N))
    # C[:N,:N] = np.diag(c1[:,0])
    # C[:N,N:] = np.diag(c2[:,0])
    # C[N:,:N] = np.diag(c2[:,0])

    # Cinv = np.zeros((2*N,2*N))
    # Cinv[N:,N:] = np.diag(-c1[:,0]/c2[:,0]**2)
    # Cinv[:N,N:] = np.diag(1/c2[:,0])
    # Cinv[N:,:N] = np.diag(1/c2[:,0])

    # print(np.linalg.det(C), np.prod(c2**2))

    # plt.figure()
    # plt.imshow(C)

    # plt.figure()
    # plt.imshow(W.C)

    # plt.figure()
    # plt.imshow(np.linalg.inv(C))

    # plt.figure()
    # plt.imshow(W.Cinv)

    # plt.show()
