"""
HMC and GPG-HMC on synthetic Bayesian linear regression
Records times and acceptance rates for multiple settngs
"""

import numpy as np
from matplotlib import pyplot as plt
import time

from linlogreg import Logistic, generate_data
from hamiltonian_mc import HamiltonianMonteCarlo, GPGradientHMC

# turn off numpy multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys

sys.path.append("../../src")
from inference import DerivativeGaussianProcess
from kernels import RBF, ExponentialKernel, LinearKernel


def toy_logreg_hmc(n, dim, seed, N_hmc):

    # Generate artificial data
    np.seed(seed)
    theta = 2*np.random.rand(dim, 1) - 1.   # Uniform[-1,1]
    X, Y = generate_data(n, theta, logistic=True)
    logreg = Logistic(X, Y, 1.)

    # HMC setup
    np.seed(seed + 42)

    x0 = np.random.randn(dim, 1)

    sqrt4_d = np.ceil(dim ** 0.25)
    T = int(12 * sqrt4_d)
    h = 1.e-2 / sqrt4_d

    N_burn = dim
    N_sample = N_hmc

    # run and time plain HMC
    hmc = HamiltonianMonteCarlo(logreg.energy, logreg.gradient, T, h, N_burn=N_burn)

    t_hmc = time.process_time()
    X_hmc = hmc.sample(N_sample, x0)
    t_hmc = time.process_time() - t_hmc

    acc_prob_hmc = hmc.diagnostics.acceptance_rate


# Settings for repeated runs
seeds = np.arange(10)
