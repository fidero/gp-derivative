import numpy as np
# from cycler import cycler
from itertools import cycle



class functionND:

    def __init__(self, N,noise_f_std=0.0, noise_grad_std=0.0, noise_seeds=None):
        self.noise_f_std = np.abs(noise_f_std)
        self.dim=N
        std = np.atleast_1d(noise_grad_std)

        if std.ndim == 1:
            self.noise_grad_std = np.diag(std * np.ones(self.dim))

        elif std.ndim == 2:
            self.noise_grad_std = np.linalg.cholesky(noise_grad_std)

        else:
            raise ValueError(
                'noise_grad_std must correspond to valid covariance')

        # print(noise_seeds)
        if noise_seeds is not None:
            self.noise_seeds = cycle(noise_seeds)

        self._f_noise = 0.0
        self._g_noise = np.zeros(self.dim)

    def _f(self, x):
        raise NotImplementedError()

    def _g(self, x):
        raise NotImplementedError()

    def _hp(self,x,p):
        raise NotImplementedError()

    def new_batch(self):

        # try:
        #     np.random.seed(next(self.noise_seeds))
        # except TypeError:
            # pass

        if hasattr(self, 'noise_seeds'):
            # act_seed=next(self.noise_seeds)
            # print(act_seed)
            # np.random.seed(act_seed)
            np.random.seed(next(self.noise_seeds))

        self._f_noise = np.random.randn()
        self._g_noise = np.random.randn(self.dim)

    def f(self, x):
        Z = self._f(x)
        # return Z + self.noise_f_std * np.abs(np.random.randn(*Z.shape))
        self.x = x
        return Z + self.noise_f_std * np.abs(self._f_noise)

    def g(self, x):
        grad = self._g(x)
        grad_dim = grad.ndim
        noise = self._g_noise @ self.noise_grad_std
        if grad_dim > 1:
            noise = noise.reshape((self.dim, *(-1,)*(grad_dim-1)), order='f')
        return grad + noise

    def fg(self, x):
        return self.f(x), self.g(x)


    @property
    def global_fmin(self):
        # return self._f(self.global_min)
        return [self._f(x) for x in self.global_min]

    def __call__(self, x):
        return self.f(x)

    def __repr__(self):
        rep = self.__class__.__name__ + f"({self.dim}D)"

        return rep

    def test_grad(self, eps=1e-4):
        v = np.random.randn(*self.x0.shape)
        # v /= np.linalg.norm(v)
        fplus = self._f(self.x0 + eps * v)
        fminus = self._f(self.x0 - eps * v)
        f0 = self._f(self.x0)

        df_v = (fplus - fminus) / (2 * eps)
        g0 = self._g(self.x0)
        gv = g0.dot(v)

        diff = np.abs(gv - df_v) / df_v
        if abs(diff) < eps:
            # if np.allclose(gv, df_v, rtol=eps):
            print(f'Gradient close to numerical: {diff:9.2e}, {eps:.2e} ({self.__class__.__name__})')
            # print(f'Gradient close to numerical: {diff:.2e}, {10*eps**2:.2e}')
        else:
            raise ValueError(f'Check gradient implementation: {diff:.2e}, {eps:.2e}, {np.linalg.norm(g0):.2e} ({self.__class__.__name__})')
            # raise ValueError(f'Check gradient implementation: {diff:.2e}, {10*eps**2:.2e}, {np.linalg.norm(g0):.2e}')


class function2D(functionND):
    """docstring for function2D"""
    def __init__(self, noise_f_std=0.0, noise_grad_std=0.0, noise_seeds=None):
        super(function2D, self).__init__(N=2, noise_f_std=noise_f_std, noise_grad_std=noise_grad_std, noise_seeds=noise_seeds)

    def grid_f(self, X, Y):
        Xgrid = np.vstack((X[np.newaxis, ...], Y[np.newaxis, ...]))
        return self._f(Xgrid)

    @property
    def plotlimits(self):
        x = (self.plotmin[0], self.plotmax[0])
        y = (self.plotmin[1], self.plotmax[1])
        return x, y





if __name__ == '__main__':

    N = 3
    seeds = np.random.randint(200, size=N)
    # seeds = None
    # seed_cycler = cycle(seeds)
    # print(seeds)
    # print(seed_cycler)

    c = np.random.randn(2, 2)
    c = np.linalg.cholesky(c.dot(c.T)) + np.eye(2)

    fun = function2D(noise_f_std=0.1, noise_grad_std=c, noise_seeds=seeds)
    for i in range(3 * N):
        fun.new_batch()
        print(fun._f_noise)

    print()
