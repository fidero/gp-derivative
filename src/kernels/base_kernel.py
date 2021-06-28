from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from scipy.sparse import csr_matrix, eye
from woodbury import SymmetricWoodbury



def build_C(K):
    """
    Constructs the scaled transpose matrix from provided Gram matrix.
    """
    Ma, Mb = K.shape
    C = np.zeros((Ma * Mb, Mb * Mb))

    for a in range(Ma):
        for b in range(Mb):
            C[a * Mb + b, b * Mb + a] = K[a, b]
    return C


class Kernel(ABC):
    """
    A kernel for Gaussian Process regression with derivatives
    """

    def __init__(self, input_dim: int, **params):
        """
        :param input_dim: Input dimension
        """
        self.input_dim = input_dim
        self.hyperparameters = dict(**params)
        self._num_obs = 0

    def __str__(self):
        mess=self.__class__.__name__ +f"({self.input_dim})"+':\n'
        # mess = ""
        for k, v in self.hyperparameters.items():
            mess += f"\t {k}: {np.atleast_1d(v).T}\n"
        # mess += f"observations: {self._num_obs}"
        return mess

    def update_hyperparameter(self,name,value):

        try:
            saved_val = self.hyperparameters[name]

            if type(saved_val) != type(value):
                raise TypeError(f"Wrong type: expected {type(saved_val)} but got {type(value)}")

            if type(saved_val)==np.ndarray:
                assert saved_val.shape==value.shape, f'shape mismatch for "{name}": {saved_val.shape} | {value.shape}'

            self.hyperparameters[name]=value

        except KeyError:
            raise KeyError(f'"{name}" not among valid hyperparameters')


    def __call__(self, xa, xb):
        return self.K(xa, xb)

    @abstractmethod
    def K(self, xa, xb):
        """ Evaluate the kernel """
        pass

    # @abstractmethod
    # def dK(self, xa, xb):
    #     """ Kernel derivative w.r.t. first argument """
    #     pass

    # @abstractmethod
    # def dKd(self, xa, xb):
    #     """ Kernel derivative w.r.t. first and second argument """
    #     pass

    @abstractmethod
    def _dKd_explicit(self, xa, xb):
        """ Kernel derivative w.r.t. first and second argument """
        pass

    # @abstractmethod
    # def ddKd(self, xa, xb):
    #     """
    #     Kernel Hessian w.r.t. first and derivative w.r.t. second argument
    #     """
    #     pass


class InnerProductKernel(Kernel, ABC):
    """
    Kernels that are a function of a scaled inner product, i.e. k(r)
        r = (x-c)^T W (x-c)
    where W is assumed to be a diagonal matrix
    """

    def __init__(
        self,
        input_dim: int,
        w: Union[float, np.ndarray],
        c: Union[float, np.ndarray],
        **kwargs,
    ):
        """
        :param input_dim: Input dimension
        :param w: float or np.ndarray
        :param c: np.ndarray reference point
        """
        if isinstance(w, float):
            # TODO: that's not so cool. But easy to take both floats and vectors.
            # Also:  Need to figure out how to handle vector c
            assert w > 0, f"w must be positive: {w}"

            w = w * np.ones((input_dim, 1))
        if isinstance(c, float):
            c_ = c*np.ones((input_dim, 1))

        elif isinstance(c, np.ndarray):
            c = np.atleast_2d(c)
            if c.shape == (input_dim, 1):
                c_ = np.copy(c)
            elif c.shape == (1, input_dim):
                c_ = np.copy(c.T)
            else:
                raise ValueError(f"Unknown type of c: {type(c)}")

        super().__init__(input_dim, w=w, c=c_, **kwargs)

    def change_c(self,new_c):
        old_c = self.hyperparameters["c"]
        if old_c.shape == new_c.shape:
            self.hyperparameters["c"]=new_c
        else: 
            raise TypeError("new_c ({type(new_c)}) does not match current c ({type(new_c)})")


    def _get_XTWX(self,xa,xb=None,saved=False):
        """Handle to avoid unnecessary calculation of dot products.
        """

        c = self.hyperparameters["c"]
        W = self.hyperparameters["w"]

        if xb is None:
            if not saved:
                d = xa-c
                self.XTWX = d.T.dot(W *d)

            return self.XTWX
        else:
            return (xa - c).T.dot(W * (xb - c))

    def _dK_dr(self, xa, xb):
        """
        Derivative w.r.t. r
        """
        raise NotImplementedError

    def _d2K_dr2(self, xa, xb):
        """
        Second derivative of the kernel w.r.t. r
        """
        raise NotImplementedError

    def _d3K_dr3(self, xa, xb):
        """
        Third derivative of the kernel w.r.t. r
        """
        raise NotImplementedError

    def _dKd_explicit(self, xa, xb):
        """ Explicitly built kernel derivative w.r.t. first and second argument in for loops"""
        Ma = xa.shape[1]
        Mb = xb.shape[1]
        D = self.input_dim
        Kp = self._dK_dr(xa, xb)
        Kpp = self._d2K_dr2(xa, xb)

        W = self.hyperparameters["w"]
        c = self.hyperparameters["c"]

        Wxa = W * (xa - c)
        Wxb = W * (xb - c)

        G = np.zeros((D * Ma, D * Mb))

        for a in range(Ma):
            for b in range(Mb):
                # G[a * D : (a + 1) * D, b * D : (b + 1) * D] =  Kpp[a, b] * np.outer(Wxa[:, a:a+1],Wxb[:, b:b+1].T)
                # G[a * D : (a + 1) * D, b * D : (b + 1) * D] =  Kpp[a, b] * np.outer(Wxb[:, b:b+1],Wxa[:, a:a+1].T)
                # G[a * D : (a + 1) * D, b * D : (b + 1) * D] = Kp[a, b] * np.diag(W[:,0]) + Kpp[a, b] * np.outer(Wxa[:, a],Wxb[:, b])
                G[a * D : (a + 1) * D, b * D : (b + 1) * D] = Kp[a, b] * np.diag(
                    W[:, 0]
                ) + Kpp[a, b] * np.outer(Wxb[:, b], Wxa[:, a])

        return G

    def dKd(self, xa, xb):
        """ Kernel derivative w.r.t. first and second argument as Kronecker product"""
        Ma = xa.shape[1]
        Mb = xb.shape[2]
        N = self.input_dim
        Kp = self._dK_dr(xa, xb)
        Kpp = self._d2K_dr2(xa, xb)

        W = self.hyperparameters["w"]
        c = self.hyperparameters["c"]

        Wxa = W * (xa - c)
        Wxb = W * (xb - c)

        G = np.kron(Kp, np.diag(W))

        for a in range(Ma):
            for b in range(Mb):
                G[a * N : (a + 1) * N, b * N : (b + 1) * N] = (
                    Kp[a, b] * np.diag(W) + Kpp[a, b] * np.outer(Wxb[:, b], Wxa[:, a]),
                )

        return G

    def _Kd_mv(self, xa, xb, Z):
        # TODO: test
        w = self.hyperparameters["w"]
        c = self.hyperparameters["c"]
        Kp = self._dK_dr(xa, xb)
        wZ = w * Z
        print(Kp.shape, (xa - c).shape, wZ.shape)
        return (Kp * ((xa - c).T @ wZ)).sum(axis=-1)

    def _dKd_mv(self, xa, xb, Z):
        w = self.hyperparameters["w"]
        c = self.hyperparameters["c"]
        Kp = self._dK_dr(xa, xb)
        Kpp = self._d2K_dr2(xa, xb)
        wZ = w * Z

        return wZ.dot(Kp.T) + w * (xb - c).dot(Kpp.T * (wZ.T.dot((xa - c))))

    def _ddKd_mv(self, xa, xb, Z):
        w = self.hyperparameters["w"]
        c = self.hyperparameters["c"]
        Kpp = self._d2K_dr2(xa, xb)
        Kppp = self._d3K_dr3(xa, xb)
        wX = w * (xb - c)
        wZ = w * Z
        t = (xa - c).T.dot(Z)
        D1 = t * Kppp
        D2 = Kpp
        U = np.hstack((wX,wZ))

        return SymmetricWoodbury(w * np.ravel(Kpp.dot(t.T)), U, D1, D2)

        # return (
        #     (D2 * wX).dot(wZ.T)
        #     + wZ.dot((D2 * wX).T)
        #     + (D1 * wX).dot(wX.T)
        #     + np.diag(w[:, 0] * np.ravel(Kpp.dot(t.T)))
        # )



class StationaryKernel(Kernel, ABC):
    def __init__(self, input_dim: int, w: Union[float, np.ndarray]):
        """
        :param input_dim: Input dimension
        :param w: float or np.ndarray
        """
        if isinstance(w, float):
            # TODO: that's not so cool. But easy to take both floats and vectors.
            assert w > 0, f"w must be positive: {w}"

            w = w * np.ones((input_dim, 1))

        super().__init__(input_dim, w=w)

    def _get_XTWX(self,xa,xb=None,saved=False):
        """Handle to avoid unnecessary calculation of dot products.
        """
        W = self.hyperparameters["w"]

        if xb is None:
            if not saved:
                self.XTWX=xa.T.dot(W*xa)

            d = np.diag(self.XTWX).reshape(-1, 1)
            return d + d.T - 2*self.XTWX
        else:

            XTWX = xa.T.dot(W*xb)

            D = (
                    np.sum(xa * W * xa, axis=0, keepdims=True).T
                    + np.sum(xb * W * xb, axis=0, keepdims=True)
                    - 2 * XTWX
                )
            return D

    

    def _L(self, M):
        row = np.zeros(M * M, dtype=np.int)
        col = np.zeros(M * M, dtype=np.int)
        for a in range(M):
            aM = a * M
            a1M = (a + 1) * M
            row[aM:a1M] += aM + a
            col[aM:a1M] += np.arange(aM, a1M)

        L = csr_matrix((np.ones(M * M), (row, col)), dtype=np.float) - eye(
            M * M, dtype=np.float
        )

        return L

    def _dK_dr(self, xa, xb):
        """
        Derivative w.r.t. r
        """
        # Consider using -2*dK_dr
        raise NotImplementedError

    def _d2K_dr2(self, xa, xb):
        """
        Second derivative of the kernel w.r.t. r
        """
        # Consider using 4*d2K_dr2
        raise NotImplementedError

    def _dKd_explicit(self, xa, xb):
        """ Explicitly built kernel derivative w.r.t. first and second argument in for loops"""
        Ma = xa.shape[1]
        Mb = xb.shape[1]
        N = self.input_dim
        Kp = self._dK_dr(xa, xb)
        Kpp = self._d2K_dr2(xa, xb)

        W = self.hyperparameters["w"]

        Wxa = W * xa
        Wxb = W * xb

        G = np.zeros((N * Ma, N * Mb))

        for a in range(Ma):
            for b in range(Mb):
                # G[a * N : (a + 1) * N, b * N : (b + 1) * N] =-Kp[a, b] * np.diag(W[:,0]) + Kpp[a, b]* np.outer(Wxb[:, b] - Wxa[:, a], Wxa[:, a] - Wxb[:, b])
                G[a * N : (a + 1) * N, b * N : (b + 1) * N] = -2 * Kp[a, b] * np.diag(
                    W[:, 0]
                ) - 4 * Kpp[a, b] * np.outer(
                    Wxa[:, a] - Wxb[:, b], Wxa[:, a] - Wxb[:, b]
                )

        return G

    def _Kd_mv(self, xa, xb, Z):
        w = self.hyperparameters["w"]
        Kp = self._dK_dr(xa, xb)
        wZ = w * Z
        temp = -2 * Kp * (xa.T.dot(wZ) - np.sum(wZ * xb, axis=0, keepdims=True))

        return np.sum(temp,axis=1)
        # return w * (
        #     -2.0 * Z.dot(Kp.T)
        #     + np.einsum("iab,ab->ia", xa[..., np.newaxis] - xb[:, np.newaxis, :], temp)
        # )

    def _dKd_mv(self, xa, xb, Z):
        w = self.hyperparameters["w"]
        Kp = self._dK_dr(xa, xb)
        Kpp = self._d2K_dr2(xa, xb)
        wZ = w * Z
        temp = -4 * Kpp * (xa.T.dot(wZ) - np.sum(wZ * xb, axis=0, keepdims=True))
        return w * (
            -2.0 * Z.dot(Kp.T)
            + np.einsum("iab,ab->ia", xa[..., np.newaxis] - xb[:, np.newaxis, :], temp)
        )

    def _ddKd_mv(self, xa, xb, Z):
        w = self.hyperparameters["w"]
        Kpp = self._d2K_dr2(xa, xb)
        # Kppp = self._d3K_dr3(xa, xb)
        Kppp = self._d3K_dr3(xa, xb)
        wX = w * (xa - xb)
        wZ = w * Z
        t = np.sum(wX * Z, axis=0,keepdims=True)
        D1 = -8 * t * Kppp
        # D1 = 8 * t * Kppp
        D2 = -4 * Kpp
        U = np.hstack((wX,wZ))
        # print(D1,D2)
        
        return SymmetricWoodbury(w * np.ravel(Kpp.dot(t.T)), U, D1, D2)

        # return (
        #     (D2 * wX).dot(wZ.T)
        #     + wZ.dot((D2 * wX).T)
        #     + (D1 * wX).dot(wX.T)
        #     + np.diag(w[:, 0] * np.ravel(t.dot(D2.T)))
        # )


