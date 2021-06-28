#!/usr/bin/env python3
"""
Module Docstring
"""
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib as mpl
# import tikz

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# from matplotlib.patches import Patch

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
    color=["#000082", "#820000", "#008200", "#B8860B", "#7600A1", "#006374"])
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0
mpl.rcParams['savefig.transparent'] = True
mpl.rcParams['image.cmap'] = 'RdBu_r'
# mpl.style.use('seaborn-dark-palette')
# mpl.style.use('tableau-colorblind10')

# Dusk gradient #ffd89b → #ffffff → #19547b
# #003973 #e5e5be


class Quadratic(object):

    """docstring for Quadratic"""

    def __init__(self, A, solution, noise=None, C=1):
        super(Quadratic, self).__init__()
        self.A = A
        self._x = solution

        self.noise_std = noise
        self.C = C

        # self.last_x= np.zeros_like(self._x)

    def _dist(self, x):
        diff_dim = x.ndim - self._x.ndim
        if diff_dim == 0:
            dist = x - self._x
        elif diff_dim == 1:
            dist = x - self._x[..., np.newaxis]
        elif diff_dim == 2:
            dist = x - self._x[..., np.newaxis, np.newaxis]

        return dist

    def hess_vec(self, v):

        if self.noise_std is not None:
            noise = self.noise_std * np.random.randn(*v.shape)
        else:
            noise = 0

        Hv = self.A.dot(v)
        return Hv + noise

    def phi(self, x):

        dist = self._dist(x)

        return 0.5 * np.einsum('i...,ij,j...', dist, self.A, dist) + self.C

    def dphi(self, x):

        dist = self._dist(x)
        df = np.einsum('ij,j...->i...', self.A, dist)
        if self.noise_std is not None:
            noise = self.noise_std * \
                np.linalg.norm(df) * np.random.randn(*dist.shape)
        else:
            noise = 0

        # df=np.einsum('ij,j...->i...', self.A, dist)
        return df + noise

    # def phi(self, x):
    #   d = self._dist(x)
    #   return 0.5 * d.dot(self.hess_vec(d))

    # def dphi(self, x):
    #   d = self._dist(x)
    #   return self.A.dot(d)

    def residual(self, x):
        return -self.dphi(x)


def headline(text, border="=", *, width=50):
    return f" {text} ".center(width, border)


def line_vec(f, t):
    if f.ndim == 1 and t.ndim == 1:
        line = np.hstack((f[..., np.newaxis], t[..., np.newaxis]))
    else:
        line = np.hstack((f, t))
    return line


def vector_alignment(u, v):
    u_ = np.sqrt(np.inner(u, u))
    v_ = np.sqrt(np.inner(v, v))
    cos_phi = np.inner(u, v) / (u_ * v_)
    return cos_phi


# pcolors = {
#     'HEADER': '\033[95m',
#     'OKBLUE': '\033[94m',
#     'OKGREEN': '\033[92m',
#     'WARNING': '\033[93m',
#     'FAIL': '\033[91m',
#     'ENDC': '\033[0m',
#     'BOLD': '\033[1m',
#     'UNDERLINE': '\033[4m'
# }

class pcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def printc(text, c=None, bold=False, underline=False):
        ''' Prints colored text to the terminal. 'c' corresponds to a selected color of:
          HEADER: (default) purple
          OKBLUE: blue
          OKGREEN: green
          FAIL: red
          '''

        if c is None:
            pre_text = pcolors.HEADER
        else:
            pre_text = pcolors.c
        # pre_text = pcolors.get(c)

        if bold:
            pre_text += pcolors.BOLD

        if underline:
            pre_text += pcolors.UNDERLINE

        print_text = pre_text + text + pcolors.get(ENDC)
        print(pre_text)


def unit_cartesian_vector(i, n):
    v = np.zeros((n, 1))
    v[i] = 1
    return v


def square_distance(x, y):
    '''computes the square distance between (x-y) x: N,d
    y: N,d  '''
    if x.ndim == 1:
        dist = np.sum((x - y)**2)
    else:
        dist = np.sum(x**2, axis=1, keepdims=True) + \
            np.sum(y**2, axis=1, keepdims=True).T - 2 * x.dot(y.T)
    return dist


def random_binary_vector(n, m):
    I = np.random.choice(np.arange(n), size=m, replace=False)
    S = np.zeros((n, m))
    S[I, np.arange(m)] += 1
    return S


def vectorize(A):
    return A.reshape(-1, 1)


def unvectorize(A, n):
    return A.reshape(n, -1)


def random_rotation_matrix(size, SO=True):
    n = size
    r = np.random.rand()
    t = r * 2 * np.pi
    if SO:
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    else:
        if np.random.rand() < 0.5:
            R = np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])
        else:
            R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    # print R

    for d in np.arange(2, n):
        v = np.random.randn(d + 1, 1)
        v /= v.T.dot(v)
        e = np.zeros((d + 1, 1))
        e[0] = 1
        x = (e - v) / np.sqrt((e - v).T.dot(e - v))
        D = np.vstack((unit_cartesian_vector(0, d + 1).T,
                       np.hstack((np.zeros((d, 1)), R))))
        R = D - 2 * x.dot(x.T.dot(D))

    R *= -1
    return R


def symmetric_projection(n):

    T = np.eye(n**2)
    z = 0
    for i in range(n):
        for j in range(n):
            T[z, j * n + i] += 1
            z += 1
    return 0.5 * T


def EigenVecToMatrix(eigenVec, eigenVal):
    n = np.size(eigenVal)
    A = np.zeros((n, n))
    for i in range(n):
        A += eigenVal[i] * eigenVec[:, i:i + 1].dot(eigenVec[:, i:i + 1].T)
    return A


def box_product(A, B):

    dimB, dimA = B.shape, A.shape
    box = np.zeros((dimA[0] * dimB[0], dimA[1] * dimB[1]))

    for i in range(dimA[0]):
        for j in range(dimB[0]):
            a = dimB[0] * i + j
            # A[i]=np.kron(WS.T[a],WS[m])
            box[a] = np.kron(B[j], A[i])
    return box


def ddiag(A):
    return np.diag(np.diag(A))


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def draw_wishart(n, C=None, nu=None):
    if nu is None:
        nu = n + 1

    if C is None:
        U = 1 / np.sqrt(nu) * np.random.randn(n, nu)
    else:
        U = 1 / np.sqrt(nu) * C.dot(np.random.randn(n, nu))
    return U.dot(U.T)


def cartesian_rotation_matrix(N, idx=(0, 1), angle=np.pi / 6):
    i, j = idx
    R = np.eye(N)
    R[i, i] = np.cos(angle)
    R[j, j] = np.cos(angle)
    R[i, j] = np.sin(angle)
    R[j, i] = -np.sin(angle)
    return R


def symmetric_diagonal_matrix(D, M=1):
    N = D.shape[0]
    U = np.random.randn(N, M)

    R = np.eye(N) - 2 * U.dot(np.linalg.solve(U.T.dot(U), U.T))
    H = R.dot(np.diag(D).dot(R.T))
    return H


def rotation_matrix(U, angle=np.pi / 6):
    """ Rotate in the 2-dim subspace of orthogonal vector space U: N,2
    """
    N = U.shape[0]
    r1 = np.array([[np.cos(angle), np.sin(angle)],
                   [-np.sin(angle), np.cos(angle)]])
    R = np.eye(N) + U.dot((r1 - np.eye(2)).dot(U.T))
    return R


def rotate_vector(U, v, angle=np.pi / 6):
    """ Rotate in the 2-dim subspace of orthogonal vector space U: N,2
    """
    # N = U.shape[0]

    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])

    Rv = v + U.dot((rot - np.eye(2)).dot(U.T.dot(v)))
    return Rv


def plot_single_matrix(mat, cmap='RdBu_r', center=False, **kwargs):
    fig = plt.figure(**kwargs)
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_xticks([])
    if center:
        lim = np.max(np.abs(mat))
        im = ax.imshow(mat, vmin=-lim, vmax=lim, cmap=cmap, **kwargs)
    else:
        im = ax.imshow(mat, norm=MidpointNormalize(
            midpoint=0), cmap=cmap, **kwargs)
    return im, fig


def make_error_boxes(ax, xdata, ydata, yerror, width=0.2, facecolor='r',
                     edgecolor='None', alpha=0.5):

        # Create list for all the error patches
    errorboxes = []
    if yerror.ndim == 1:
        yerror = np.vstack((yerror, yerror))

    n = xdata.shape[0]
    xerror = width * np.ones((2, n))

    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror,  # yerr=yerror,
                          fmt='None', ecolor=facecolor)

    return artists


def save_symmetric_matrix_samples(im, C, iterations=12, mu=None, V=None, start_sample=None, root='fig/togif', name='', outformat='png', save=True):

    D = C.shape[0]

    if V is None:
        V, _ = np.linalg.qr(np.random.randn(D**2, 2))
    else:
        V, _ = np.linalg.qr(V)

    if start_sample is None:
        draw_n = start_sample
    else:
        draw_n = np.random.randn(D**2, 1)
    angle = 2 * np.pi / iterations

    savename = root + name + '_{:02d}.' + outformat

    for i in range(iterations):

        # plt.savefig('fig/matrix_draw{:02d}.png'.format(i), dpi=72,bbox_inches='tight', transparent=True)

        draw_n = rotate_vector(V, draw_n, angle=angle)

        draw = C.dot(draw_n.reshape(D, D).dot(C.T))
        if mu is None:
            f = draw + draw.T
        else:
            f = mu + draw + draw.T

        im.set_data(f)
        # plt.imsave('fig/{:02d}.png'.format(i),f,vmin=-h,vmax=h,cmap='RdBu_r')
        if save:
            plt.savefig(savename.format(i),
                        dpi=72, bbox_inches='tight', transparent=True)
        else:
            plt.draw()
            plt.pause(0.3)

def plot_matrices(matrices, titles=None, cmap='RdBu_r', verbose=True, normalize=False, **kwargs):
    """
    Return fig, axarr
    """
    N = len(matrices)

    if titles is None:

        titles = []
        for mat in matrices:
            titles.append('%3.3f' % (np.max(np.abs(mat))))

        # titles = [""] * N
    all_min = []
    all_max = []
    widths = []
    for mat in matrices:
        widths.append(mat.shape[1])
        all_min.append(np.min(mat))
        all_max.append(np.max(mat))

    all_min = min(all_min)
    all_max = max(all_max)

    # cmap = mpl.cm.get_cmap(cmap)
    # new_cm =
    fig, axarr = plt.subplots(
        1, N, gridspec_kw={'width_ratios': widths}, **kwargs)  # gridspec)

    if N == 1:
        axarr.imshow(matrices[0], norm=MidpointNormalize(
            midpoint=0), cmap=cmap)
        # axarr.set_title('%3.3f' % (np.max(np.abs(matrices[0]))))
        if verbose:
            axarr.set_title(titles[0])
        axarr.set_xticks([])
        axarr.set_yticks([])
    else:
        for i in np.arange(N):
            if normalize:
                axarr[i].imshow(
                    matrices[i], vmin=all_min, vmax=all_max, norm=MidpointNormalize(midpoint=0), cmap=cmap)
            else:
                axarr[i].imshow(
                    matrices[i], norm=MidpointNormalize(midpoint=0), cmap=cmap)

            # axarr[i].set_title('%3.3f' % (np.max(np.abs(matrices[i]))))
            if verbose:
                axarr[i].set_title(titles[i])
            axarr[i].set_xticks([])
            axarr[i].set_yticks([])

    # plt.show()
    return fig, axarr


def tikz_raster(data, filename, FIG_WIDTH=345.0 / 72.27, **kwargs):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(FIG_WIDTH, FIG_WIDTH * height / width, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, **kwargs)
    # Choose filetype
    plt.savefig('{}1.png'.format(filename))
    # plt.savefig('{}1.pdf'.format(filename))

################
# Usage
################


# filename = 'fig/test'
# plotkwargs = {'cmap': cm.RdBu, 'vmin': abs(Z).min(), 'vmax': abs(
#     Z).max(), 'extent': [0, 2*np.pi, 0, 2*np.pi], 'interpolation': 'bilinear'}
# tikzkwargs = {'figureheight': '\\figheight',
#               'figurewidth': '\\figwidth', 'tex_relative_path_to_data': 'fig'}

##################
# Additional plotting
#################
# plt.imshow(Z, **plotkwargs)
# plt.plot(Xdata[:, 0], Xdata[:, 1], 'ko', markerSize=2)
# plt.xticks((1+np.arange(2))*np.pi, [r'$\pi$', r'2$\pi$'])
# plt.yticks((1+np.arange(2))*np.pi, [r'$\pi$', r'2$\pi$'])


# Save
# tikz_save(filepath='{}.tex'.format(filename), **tikzkwargs)
# tikz_raster(Z,filename,**plotkwargs)


def generate_spectrum(values, position=None, N=100, frac=False):
    '''
    Function to generate eigenvalues that are linearly interpolated between values at position.
    values: iterable of ranges to linealy interpolate
    position: positions to insert values. If None evenly ditributed 
    N: Length of returned array
    frac: Boolean indicates if position is given as fraction of N or integer locations  
    '''





    if position == None:
        position = np.linspace(start=0,stop=N,num=len(values)+1,dtype=np.int)
    else:
        if frac:
            position=[int(p*N) for p in position]
        if len(position) != len(values)+1:
            # sys.exit("position length must be the same as values")
            raise ValueError("Length of position is length of values + 1")
        elif position[0] != 0 or position[-1] != N:
            raise ValueError("position[0] = 0 and position[-1] = N")

    arr=np.zeros(N)

    for p,val in enumerate(values):
        p1=position[p]
        p2=position[p+1]
        arr[p1:p2]=np.linspace(val[0],val[1],num=p2-p1)

    return arr