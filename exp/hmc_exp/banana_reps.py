import sys
import argparse
import numpy as np
import pandas as pd
import datetime

sys.path += ["../../src", "../"]
from test_functions.banana import Banana
from hamiltonian_mc import HamiltonianMonteCarlo, GPGradientHMC
from inference import DerivativeGaussianProcess
from kernels import RBF


def run_banana(D, N_samples, f_seed=None, seed=None, verbose=True):
    """
    Run HMC on Banana
    :param D: dimension
    :param N_samples: number of samples
    :param f_seed: seed for function setup
    :param seed: seed for HMC run
    """

    # set up standard problem
    a = np.ones((D+1)) * 2.
    a[2] = -a[2]
    f = Banana(dim=D, a=a, seed=f_seed)

    # HMC parameters (preset)
    sqrt4_d = np.ceil(D ** 0.25)
    T = 32 * int(sqrt4_d)
    h = 2.e-3 / sqrt4_d

    N_burn = D

    # RUN HAMILTONIAN MONTE CARLO
    # ---------------------------
    if verbose:
        print("Running HMC sampling...")
    np.random.seed(seed)
    x0 = np.random.randn(D, 1)

    hmc = HamiltonianMonteCarlo(E=f.pot_energy, gradE=f.grad_e, T=T, h=h, N_burn=N_burn)
    X_hmc = hmc.sample(N_samples=N_samples, x0=x0)

    prob_hmc = hmc.diagnostics.acceptance_rate
    if verbose:
        print(f"done! HMC acceptance rate is {prob_hmc:.10f}")

    # RUN GP GRADIENT HAMILTONIAN MONTE CARLO
    # ---------------------------------------
    if verbose:
        print("Running GPG-HMC sampling...")
    # reset starting point and seed
    np.random.seed(seed)
    x0 = np.random.randn(D, 1)

    kern = RBF(D, 4. / D)
    gp = DerivativeGaussianProcess(kern)

    gpg_hmc = GPGradientHMC(
        E=f.pot_energy,
        gradE=f.grad_e,
        gp_model=gp,
        T=T,
        h=h,
        N_train=int(np.sqrt(D)),
        N_burn=N_burn,
    )
    X_gph = gpg_hmc.sample(N_samples=N_samples, x0=x0)

    prob_train = gpg_hmc.diagnostics.acceptance_rate
    prob_gph = gpg_hmc.gp_trainer.diagnostics.acceptance_rate

    if verbose:
        print(f"done! GPG-HMC acceptance rate is {prob_train:.10f} during training"
              f" and {prob_gph:.10f} on the surrogate.")

    n_gp = gpg_hmc.gp_trainer.N_dat
    n_hmc = None
    if 'N_hmc' in gpg_hmc.diagnostics.info.keys():
        n_hmc = gpg_hmc.diagnostics.info['N_hmc'][0]

    return prob_hmc, prob_train, prob_gph, n_gp, n_hmc


if __name__ == "__main__":

    # Repeat banana experiment for different seeds
    seeds = np.arange(10)
    f_seeds = seeds + 42

    data = []

    for f in f_seeds:
        print(f'Banana configuration #{f-41}')
        for s in seeds:
            result = run_banana(100, 2000, f, s, verbose=False)
            data.append([f, s] + list(result))

    df = pd.DataFrame(data, columns=['f', 'seed', 'p_hmc', 'p_train', 'p_gp', 'n_gp', 'n_hmc'])

    # save runs
    timestamp = datetime.date.today().strftime("%y%m%d")
    outfile = '../../out/hmc/banana_reps_' + timestamp
    df.to_csv(outfile, index=False)


    # Parsing arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--fun", metavar="f", type=str, default="banana"
    # )  # function ['banana']
    # parser.add_argument("--dim", metavar="d", type=int)  # Dimension
    # parser.add_argument(
    #     "--fun-seed", metavar="fs", type=int, default=None
    # )  # Seed for function setup (optional)
    # parser.add_argument(
    #     "--seed", metavar="s", type=int, default=None
    # )  # Seed for HMC (optional)
    # parser.add_argument("--samples", metavar="n", type=int)  # Number of HMC samples
    # parser.add_argument(
    #     "--kern", metavar="K", type=str, default="rbf"
    # )  # choose kernel ['rbf', 'linear', 'exp']
    # parser.add_argument(
    #     "--savedir",
    #     metavar="out",
    #     type=str,
    #     default="../out/hmc/",
    # )
    #
    # args = parser.parse_args()

    # un experiment

