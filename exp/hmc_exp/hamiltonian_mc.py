import numpy as np


class HamiltonianMonteCarlo:
    """ Hamiltonian Monte Carlo """

    def __init__(self, E, gradE, T, h, mass=1., N_burn=0, info=True):
        """
        :param E: Callable, - log P where P is the density to sample from
        :param gradE: Callable, gradient of E
        :param T: number of leapfrog steps
        :param h: stepsize for leapfrog
        :param mass: mass of fictitious particle (kin. energy is p^T p / 2m)
        :param N_burn: int, number of samples drawn for burn-in
        :param info: boolean, whether to collect diagnostics
        """
        self.E = E
        self.gradE = gradE
        self.T = T
        self.h = h
        self.m = mass
        self.N_burn = N_burn

        self.info = info

        if info:
            self.diagnostics = HMCTracker()

    def leapfrog(self, x, p, g):
        """
        Runs one leapfrog simulation of Hamiltonian Monte Carlo
        :param x: initial location
        :param p: initial momentum
        :param g: initial gradient of E
        :returns: new proposal x, new momentum p, new gradient g
        """
        for t in range(self.T):
            p = p - self.h / 2 * g
            x = x + self.h * p / self.m
            g = self.gradE(x)
            p = p - self.h / 2 * g
        return x, p, g

    def step(self, x, g, e, record_probs=True):
        """
        Propose one new state using Hamiltonian Monte Carlo
        :param x: initial location
        :param g: initial gradient of E
        :param e: initial energy E at x
        :param record_probs: whether the step is counted towards acceptance probabilities (set to False during burn-in)
        :returns: new proposal x, new gradient g, new energy e
        """
        p = np.sqrt(self.m) * np.random.randn(x.shape[0], 1)
        H = np.float(p.T @ p / (2 * self.m) + e)
        x_new, p, g_new = self.leapfrog(x, p, g)

        e_new = self.E(x_new)
        H_new = np.float(p.T @ p / (2 * self.m)  + e_new)
        dH = H_new - H

        if self.info:
            accept, accept_prob = _hmc_accept(dH)
            if record_probs:
                self.diagnostics.update_probs(prob=accept_prob, accept=accept)
            self.diagnostics.save('dH', dH)
            self.diagnostics.save('H', H_new)
            # self.diagnostics.save('accept', accept)
        else:
            accept, _ = _hmc_accept(dH)

        if accept:
            return x_new, g_new, e_new   # return new state
        return x, g, e   # return old state

    def sample(self, N_samples, x0):
        """
        Draw samples using HMC
        :param N_samples: Number of samples to draw
        :param x0: initial location, shape (D, 1)
        """
        X = np.zeros([N_samples, x0.shape[0]])
        X[0, :] = x0[:, 0]

        x = x0
        g = self.gradE(x)
        e = self.E(x)

        # burn in
        for n in range(self.N_burn):
            x, g, e = self.step(x, g, e, record_probs=False)

        # actual sampling
        for n in range(N_samples):
            x, g, e = self.step(x, g, e)
            X[n, :] = x[:, 0]

        return X

    def __call__(self, N_samples):
        return self.sample()


class GPGradientHMC:
    """ Hamiltonian Monte Carlo with GP surrogate for gradient """

    def __init__(self, E, gradE, gp_model, T, h, N_train, mass=1., N_burn=0, info=True):
        """
        :param E: Callable, - log P where P is the density to sample from
        :param gradE: Callable, gradient of E
        :param gp_model: Gaussian Process derivative model
        :param T: number of leapfrog steps
        :param h: stepsize for leapfrog
        :param N_train: int, MAXIMUM number of samples used for training of the GP
        :param mass: mass of fictitious particle (kin. energy is p^T p / 2m)
        :param N_burn: int, number of samples drawn for burn-in
        :param info: boolean, whether to collect diagnostics
        """

        self.E = E
        self.gradE = gradE
        self.gp = gp_model
        self.T = T
        self.h = h
        self.m = mass
        self.N_train = N_train
        self.N_burn = N_burn

        self.gp_trainer = GPTrainer(self.gp, self.N_train)
        self.info = info

        if info:
            self.diagnostics = HMCTracker()

    def leapfrog(self, x, p, g):
        """
        Runs one leapfrog simulation of Hamiltonian Monte Carlo
        :param x: initial location
        :param p: initial momentum
        :param g: initial gradient of E
        :returns: new proposal x, new momentum p, new gradient g
        """
        for t in range(self.T):
            p = p - self.h / 2 * g
            x = x + self.h * p / self.m
            g = self._g(x)
            p = p - self.h / 2 * g
        return x, p, g

    def step(self, x, g, e, record_probs=True):
        """
        Propose one new state using Hamiltonian Monte Carlo
        :param x: initial location
        :param g: initial gradient of E
        :param e: initial energy E at x
        :param record_probs: whether the step is counted towards acceptance probabilities (set to False during burn-in)
        :returns: new proposal x, new gradient g, new energy e
        """
        p = np.sqrt(self.m) * np.random.randn(x.shape[0], 1)
        H = np.float(p.T @ p / (2 * self.m) + e)
        x_new, p, g_new = self.leapfrog(x, p, g)

        e_new = self.E(x_new)
        H_new = np.float(p.T @ p / (2 * self.m) + e_new)
        dH = H_new - H

        if self.info:
            accept, accept_prob = _hmc_accept(dH)
            if record_probs:
                if self.gp_trainer.mode_gp:
                    self.gp_trainer.diagnostics.update_probs(prob=accept_prob, accept=accept)
                else:
                    self.diagnostics.update_probs(prob=accept_prob, accept=accept)
            self.diagnostics.save('dH', dH)
            self.diagnostics.save('H', H_new)
            # self.diagnostics.save('accept', accept)
        else:
            accept, _ = _hmc_accept(dH)

        if accept:
            return x_new, g_new, e_new   # return new state
        return x, g, e   # return old state

    def sample(self, N_samples, x0):
        """
        Draw samples using HMC
        :param N_samples: Number of samples to draw (excludes burn-in)
        :param x0: initial location
        """
        X = np.zeros([N_samples, x0.shape[0]])

        x = x0
        g = self._g(x)
        e = self.E(x)

        # burn in
        self.gp_trainer.use_grad()
        for n in range(self.N_burn):
            x, g, e = self.step(x, g, e)

        # condition GP initially
        if self.gp_trainer.N_dat < self.N_train:
            self.gp_trainer.update_gp(x, g)
        # self.gp_trainer.use_gp()

        # actual sampling
        for n in range(N_samples):

            # draw new sample
            x, g, e = self.step(x, g, e)

            if self.gp_trainer.train(x):
                # check if gp_trainer says we should train
                if not self.gp_trainer.mode_gp:
                    g = self.gradE(x)
                self.gp_trainer.update_gp(x, g)

                # Switch to using the GP after a few training points have been collected
                if self.gp_trainer.N_dat == self.N_train//2 and self.gp_trainer.mode_gp is False:
                    self.diagnostics.save('N_hmc', n)
                    self.gp_trainer.use_gp()

            X[n, :] = x[:, 0]

        return X

    def __call__(self, N_samples, x0):
        return self.sample(N_samples, x0)

    def _g(self, x):
        """
        Gradient function to be used (true gradient or surrogate)
        """
        if self.gp_trainer.mode_gp:
            return self.gp.infer_g(x)   # infer the gradient
        return self.gradE(x)


class GPTrainer:
    """
    Takes care of checking when to train the GP
    """
    def __init__(self, gp_model, N_train=None):
        self.gp = gp_model
        self.N_train = N_train
        if N_train is None:
            self.N_train = self.gp.k.input_dim

        self.N_dat = self.gp.data['dX'].shape[1] if 'dX' in self.gp.data.keys() else 0

        # start with using actual objective
        self.mode_gp = False

        # also add a tracker
        self.diagnostics = HMCTracker()

    def use_gp(self):
        self.mode_gp = True

    def use_grad(self):
        self.mode_gp = False

    def update_gp(self, x, g):
        self.N_dat +=1
        self.gp.update_data(dX=x, dY=g)

    def train(self, x):
        """ Should we train the GP? """
        if self.N_dat < self.N_train:
            return self._far_from_dat(x)
        return False

    def _far_from_dat(self, x):
        """
        Computes if the Mahalonobis distance of new x to previous data is larger than the lengthscale of the kernel
        :param x: proposal location
        """
        if self.N_dat == 0:
            return True

        w = self.gp.k.hyperparameters["w"]
        if (w == w[0]).all():
            return ~((np.linalg.norm(x - self.gp.data['dX'], axis=0) < np.sqrt(1./w[0])).any())
        else:
            raise NotImplementedError
        # TODO: extend to non-scalar lengthscale


class HMCTracker:
    """
    Tracks information about HMC, e.g. acceptance probability, timing
    """
    def __init__(self):
        self.probability_accept = 0.   # probability of acceptance averaged over HMC run
        self.acceptance_rate = 0.   # empirical acceptance rate
        self.iter_count = 0   # counting iterations
        self.info = {}

    def update_probs(self, prob, accept):
        """
        :param prob: Probability of acceptance
        :param accept: boolean, whether proposal has been accepted
        """
        self.probability_accept = self._next_prob(self.probability_accept, prob, self.iter_count)
        self.acceptance_rate = self._next_prob(self.acceptance_rate, accept, self.iter_count)
        self.iter_count += 1

    def save(self, name, value):
        """
        Enables saving arbitrary information from the HMC run into a dictionary
        """
        if name in self.info.keys():
            self.info[name] += [value]
        else:
            self.info[name] = [value]

    @staticmethod
    def _next_prob(p_mean, p_new, n):
        """
        New average when one new item is added
        """
        return n / (n+1) * p_mean + p_new / (n + 1)


# HMC utils
def _hmc_accept(dH):
    """
    Compute Metropolis acceptance criterion for HMC

    :param dH: change of Hamiltonian
    :return: accept (bool), acceptance probability
    """
    accept = False
    accept_prop = 1 if dH < 0 else np.exp(-dH)
    if dH < 0 or np.random.rand() < accept_prop:  # accept
        accept=True
    return accept, accept_prop
