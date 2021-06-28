#!/usr/bin/env python3
"""
Module Docstring
"""

# Imports
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.style.use(["icml21.mplstyle","fullpage.mplstyle"])
import sys
sys.path.append("../src")
from kernels import ExponentialKernel,build_C,RBF
from utils import plot_matrices


#############################

def main(args):
    print(args)
    np.random.seed(43)
    D = 10
    M = 3

    c = np.ones((D,1)) 

    X = 3*np.random.randn(D,M)/(D) 
    X_= X-c

    U=np.kron(np.eye(M),X_)

    w = np.ones((D,1))#np.sqrt(D)
    # w = 1+ np.abs(np.random.randn(D,1))#np.sqrt(D)

    # kern=ExponentialKernel(D,w=w,c=c)


    kern=RBF(D,w=w)
    # U=np.kron(np.eye(M),X)
    L=kern._L(M)
    U=(L.T@U.T).T

    dKd=kern._dKd_explicit(X,X)
    Kp = -2*kern._dK_dr(X,X)
    Kpp = -4*kern._d2K_dr2(X,X)
    C = build_C(Kpp)
    w_=np.diag(w[:,0])

    fig,axarr=plot_matrices([dKd,Kp,w_,U,C,U.T],titles=[""]*6,normalize=True)
    # fig,axarr=plot_matrices([dKd,Kp,w_,U,-C,U.T],titles=[""]*6)
    # fig.set_figheight(2)





    if args.save:
        filename='decomposition_switched.svg'
        savename=args.figdir+filename
        print(f"Saving figure in:{savename}")
        plt.savefig(savename)
    else:
        plt.show()


if (__name__ == "__main__"):
    print('Executing as standalone script')
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Optional float
    parser.add_argument("-lw", "--linewidth", metavar='L',
                        type=float, default=5.5)
    # Optional float
    parser.add_argument("--width", metavar='W', type=float, default=1.0)
    # Optional float
    parser.add_argument("--height", metavar='H',
                        type=float)

    # parser.add_argument("--style", metavar='S', default=None,
    #                     choices=['icml21', 'presentation'])

    parser.add_argument('--figdir', type=str, default='../fig/',
                    help='Path to save location of figures')

    # Optional argument flag which defaults to False
    parser.add_argument("-s", "--save", action="store_true", default=False)

    # Optional choices
    # parser.add_argument('-o','--optimizer', choices=['bfgs', 'dfp'],default='dfp')

    args = parser.parse_args()
    main(args)
