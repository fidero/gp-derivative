# Create a plot of a sliced 100d Hamiltonian Monte Carlo run

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path += ["../../src", "../"]
from test_functions.banana import Banana
from hamiltonian_mc import HamiltonianMonteCarlo, GPGradientHMC
from inference import DerivativeGaussianProcess
from kernels import RBF



D = 100
f_seed = None
seed = 2

# set up problem
a = np.ones((D + 1)) * 2.
a[2] = -a[2]
f = Banana(dim=D, a=a, seed=f_seed)

# HMC parameters (preset)
sqrt4_d = np.ceil(D**0.25)

T = 32 * int(sqrt4_d)
h = 4.e-3 / sqrt4_d

N_burn = D
N_samples = 2000

np.random.seed(seed)
x0 = np.random.randn(D, 1)

# RUN HAMILTONIAN MONTE CARLO
# ---------------------------
hmc = HamiltonianMonteCarlo(E=f.pot_energy, gradE=f.grad_e, T=T, h=h, N_burn=N_burn)
X_hmc = hmc.sample(N_samples=N_samples, x0=x0)
print(f"HMC acceptance rate {hmc.diagnostics.acceptance_rate:.10f}")

# RUN GP GRADIENT HAMILTONIAN MONTE CARLO
# ---------------------------------------
# reset starting point and seed
np.random.seed(seed)
x0 = np.random.randn(D, 1)

kern = RBF(D, 2.5 / D)

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
print(f"GPG-HMC acceptance rate during HMC {gpg_hmc.diagnostics.acceptance_rate:.10f}")
print(f"GPG-HMC acceptance rate during surrogate {gpg_hmc.gp_trainer.diagnostics.acceptance_rate:.10f}")
print(f"GPG-HMC draws {gpg_hmc.diagnostics.info['N_hmc']} with HMC before using GP model")
# print(f"GPG-HMC draws {gpg_hmc.gp_trainer.N_dat} with HMC before using GP model")


# Make figure
plt.style.use("icml21.mplstyle")
N_plot = 100

xlim = 2.2
ylim = 2.8
yoffset = - 0.5

x1_plot = np.linspace(-xlim, xlim, N_plot)
x2_plot = np.linspace(-ylim, ylim, N_plot) + yoffset
X1, X2 = np.meshgrid(x1_plot, x2_plot, indexing='ij')
X = (np.stack([X1, X2], axis=0)).reshape(2, -1)

X_dim = np.zeros((D, X.shape[1])); X_dim[:2,:] = X

functions = [f.f, f.f, lambda x: np.exp(-gp.infer_f(x))]
labels = [r'\textsc{hmc}', r'\textsc{gpg-hmc}']


# FIGURE
fig, axes = plt.subplots(1, 2)

# getting cmaps right
levels = 7
color_dic = {}
for color in ['Greens', 'Greys']:
    cmap = plt.cm.get_cmap(color)
    cl = []
    for v in np.linspace(0, 1, levels):
        cl += [cmap(v)]
    cl[0] = (1., 1., 1., 1.)
    color_dic[color] = cl

for i, ax in enumerate(axes):
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    ax.patch.set_visible(False)
    ax.axis('off')
    cmap = mpl.colors.ListedColormap(color_dic['Greys'])
    cpl = ax.contourf(X1, X2, f.f(X_dim).reshape(N_plot, N_plot), cmap=cmap, zorder=0)
    # cpl = ax.contourf(X1, X2, f.f(X_dim).reshape(N_plot, N_plot), cmap='Greys', levels=7, zorder=0)

    if i == 0:
        ax.scatter(X_hmc[:, 0], X_hmc[:, 1], c='C2', s=1., edgecolors='none', alpha=0.7, zorder=2)
    else:
        exp_f = np.exp(-gp.infer_f(X_dim))
        # cmap = mpl.colors.ListedColormap(color_dic['Greens'])
        ax.contour(X1, X2, exp_f.reshape(N_plot, N_plot), #colors='C1',
                    # cmap='Greens', vmin=0.01, vmax=max(exp_f), levels=6,
                    cmap = 'Greens',
                   linewidths=0.8,
                    alpha=0.7, zorder=1)
        ax.scatter(X_gph[:, 0], X_gph[:, 1], c='C2', s=1., edgecolors='none', alpha=0.7, zorder=2)
        ax.scatter(gp.data['dX'][0, :], gp.data['dX'][1, :], c='yellowgreen', marker='*', s=30, edgecolors='none',
                   linewidths=0.1, zorder=3)
    ax.set_aspect('equal')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim + yoffset, ylim + yoffset)
    # plt.colorbar(cpl, ax=ax[i])
    ax.text(0.5, 0.01, labels[i], verticalalignment='bottom', horizontalalignment='center',
            transform=ax.transAxes, fontsize='small')

plt.subplots_adjust(wspace=0.1)
plt.savefig('../../fig/hmc_banana.pdf', format='pdf', bbox_inches='tight', pad_inches=0)


# functions = [f.f, f.f, lambda x: np.exp(-gp.infer_f(x))]
# labels = [r'\textsc{HMC}', r'\textsc{GPG-HMC}', r'\textsc{GPG-HMC}']

# fig, axes = plt.subplots(1, 3)
#
# for i, ax in enumerate(axes):
#     frame = plt.gca()
#     frame.axes.get_xaxis().set_visible(False)
#     frame.axes.get_yaxis().set_visible(False)
#     ax.patch.set_visible(False)
#     ax.axis('off')
#     cpl = ax.contourf(X1, X2, functions[i](X_dim).reshape(N_plot, N_plot), cmap='Greys', levels=8)
#
#     if i == 0:
#         ax.scatter(X_hmc[:, 0], X_hmc[:, 1], c='C0', s=0.5, edgecolors='none', alpha=0.6)
#     else:
#         ax.scatter(X_gph[:, 0], X_gph[:, 1], c='C0', s=0.5, edgecolors='none', alpha=0.6)
#         ax.scatter(gp.data['dX'][0, :], gp.data['dX'][1, :], c='yellowgreen', marker='*', s=16, edgecolors='none',
#                    linewidths=0.1)
#     ax.set_aspect('equal')
#     ax.set_xlim(-xlim, xlim)
#     ax.set_ylim(-ylim + yoffset, ylim + yoffset)
#     # plt.colorbar(cpl, ax=ax[i])
#     ax.text(0.95, 0.01, labels[i], verticalalignment='bottom', horizontalalignment='right',
#             transform=ax.transAxes, fontsize='xx-small')
#
# plt.subplots_adjust(wspace=0.1)
# plt.savefig('../../fig/hmc_banana.pdf', format='pdf', bbox_inches='tight', pad_inches=0)