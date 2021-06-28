import numpy as np
from scipy.linalg import cho_factor, cho_solve
import scipy.linalg as sla

# from ... import StationaryKernel, InnerProductKernel, build_C
# from .. import kernels #StationaryKernel, InnerProductKernel, build_C
# from .. import kernelsStationaryKernel, InnerProductKernel, build_C
from kernels import StationaryKernel, InnerProductKernel, build_C, FastLinearKernel


class DerivativeGaussianProcess:
    """
    Class for Gaussian Process Regression that allows to condition on function evaluations and gradients, and perform
    inference on function values, gradients, and Hessians.
    In particular, this class does not require locations for function evaluations and derivatives to coincide.
    """

    def __init__(self, kernel, noise_variance=0.0, nugget=1e-6):
        """
        :param kernel: Covariance function
        :param noise_variance: noise level of function evaluations
        TODO: There should prabably be a separate noise level for the gradients
        """
        self.k = kernel
        self.noise_variance = noise_variance
        self.nugget = nugget  # nugget is so far unused
        self.hyperparameters = {
            **self.k.hyperparameters,
            "noise_variance": self.noise_variance,
        }
        self.data = {}

    def condition(self, dX=None, dY=None):
        """
        Condition GP on given data
        Precomputes required quantities for inference.
        :param dX: locations at which there are derivative observations
        :param dY: derivative data #TODO: which format? matrix or stacked?
        -----------
        Solves GZ=dY
        """

        ############
        # Todo: include prior mean of gradient?
        ############

        D, N = dX.shape
        sig2 = self.hyperparameters["noise_variance"] + self.nugget
        w = self.hyperparameters["w"]

        self.data["dX"] = dX
        self.data["dY"] = dY

        if issubclass(type(self.k), FastLinearKernel):

            # Kpinv = np.linalg.inv(self.k._dK_dr(dX,dX) + sig2 * np.eye(N, N))
            L = cho_factor(self.k._dK_dr(dX,dX) + sig2 * np.eye(N, N),lower=True)

            c = self.k.hyperparameters["c"]
            dX_=dX-c
            # a = Kpinv@(dX_.T.dot(dY))
            if self.k.hyperparameters["p"]==1:
                self.Z = cho_solve(L, (dY/w).T).T
            else:
                a = cho_solve(L, (cho_solve(L, dX_.T.dot(dY)).T))
                # self.Z = 1 / w * dY.dot(Kpinv) - dX_.dot(a.dot(Kpinv))
                self.Z = cho_solve(L, (dY/w).T).T - dX_.dot(a)

        else:
            if N > D:
                # if N >= D:
                G = self.k._dKd_explicit(dX, dX)
                self.cho = cho_factor(G + sig2 * np.eye(N * D))
                self.Z = cho_solve(self.cho, dY.reshape(-1, 1, order="f")).reshape(
                    *dY.shape, order="f"
                )
            else:

                Kp = self.k._dK_dr(dX, dX)
                Kpp = self.k._d2K_dr2(dX, dX)
                # XTWX = dX.T.dot(w*dX)
                XTWX = self.k.XTWX
                self.data["XTWX"] = XTWX
                # print(XTWX)
                if issubclass(type(self.k), StationaryKernel):
                    # Kpinv = np.linalg.inv(-2*self.k._dK_dr(dX,dX) + sig2 * np.eye(N, N))
                    Kpinv = np.linalg.inv(-2 * Kp + sig2 * np.eye(N, N))

                    # Kpp = 4*self.k._d2K_dr2(dX,dX)
                    Kpp *= 4

                    # -2Kp x w + ULC[4Kpp]L.T U.T
                    L = self.k._L(N)
                    # TODO: 1./Kpp can contain np.infs :( --> FIX!
                    T = build_C(1.0 / Kpp) + L.T @ (np.kron(Kpinv, XTWX) @ L)
                    xtgk = L.T @ (dX.T.dot(dY).dot(Kpinv)).reshape(-1, 1, order="f")
                    a = (
                        (
                            L
                            @ sla.solve(
                                T, xtgk.reshape(-1, 1, order="f"), assume_a="sym"
                            )
                        )
                        .reshape(N, N, order="f")
                        .dot(Kpinv)
                    )
                    self.Z = 1 / w * dY.dot(Kpinv) - dX.dot(a)
                elif issubclass(type(self.k), InnerProductKernel):
                    #####
                    # Figure out how to handle c
                    #######
                    c = self.k.hyperparameters["c"]
                    dX_ = dX - c
                    Kpinv = np.linalg.inv(Kp + sig2 * np.eye(N, N))

                    Kpp = self.k._d2K_dr2(dX, dX)
                    T = build_C(1 / Kpp) + np.kron(Kpinv, XTWX)
                    # print(Kpp,Kpinv,Kp)
                    a = sla.solve(
                        T,
                        (dX_.T.dot(dY)).dot(Kpinv).reshape(-1, 1, order="f"),
                        assume_a="sym",
                    ).reshape(N, N, order="f")
                    self.Z = 1 / w * dY.dot(Kpinv) - dX_.dot(a.dot(Kpinv))

                else:
                    raise ValueError("Kernel structure not recognized")

    def update_data(self, dX=None, dY=None):
        """
        Update data of GP model, e.g. when data is added iteratively.
        :param dX: locations at which there are derivative observations
        :param dY: derivative data #TODO: which format? matrix or stacked?
        """
        # TODO: Will there be some low-rank update?
        if "dX" in self.data:
            self.data["dX"] = np.hstack([self.data["dX"], dX])
            self.data["dY"] = np.hstack([self.data["dY"], dY])
        else:
            self.data["dX"] = dX
            self.data["dY"] = dY

        # Condition GP model on current data
        self.condition(dX=self.data["dX"], dY=self.data["dY"])

    def infer_f(self, x):
        """
        Infer function values at new inputs x
        :param x: test inputs
        :returns: Posterior GP mean and covariance
        """
        # raise NotImplementedError
        return self.k._Kd_mv(x, self.data["dX"], self.Z)

    def infer_g(self, x):
        """
        Infer gradient values at new inputs x
        """
        # raise NotImplementedError
        return self.k._dKd_mv(x, self.data["dX"], self.Z)

    def infer_h(self, x):
        """
        Infer Hessian values at new inputs x
        """
        # raise NotImplementedError
        return self.k._ddKd_mv(x, self.data["dX"], self.Z)

    def neg_log_marginal_likelihood(self):
        """
        Negative log marginal likelihood of the GP
        """
        raise NotImplementedError

    def optimize_hyperparameters(self):
        """
        Optimize the hyperparameters of model
        """
        raise NotImplementedError
