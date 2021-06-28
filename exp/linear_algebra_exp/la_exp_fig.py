#!/usr/bin/env python3
##################################
# Naive implementation of optimization routines with derivative observations for linear algebra.
# 
#
#   --- Code only intended as a proof-of-concept ---
#
##################################

# Imports
import argparse
import numpy as np
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
sys.path+=["../","../../src"]


from LinearAlgebra import CG,GradientMin,HessianMin,GradientMinFixed
import kernels
from test_functions.functionsND import byNumber
from utils import random_rotation_matrix, generate_spectrum
# Global variables

# Class declarations
class RunInformation:
    
    def __init__(self,fun,der,**params):
        self._f=fun
        self._g=der
        self._x=[]
        self._gnorm=[]
        self._fval=[]
        self._fname=fun.__class__.__name__
        self._params={**params}
        
    def save_data(self,xk):
        self._x.append(np.copy(xk))
        self._fval.append(self._f(xk))
        self._gnorm.append(np.linalg.norm(self._g(xk)))


    @property
    def fval(self):
        return np.array(self._fval)

    @property
    def x(self):
        return np.array(self._x)

    @property
    def gnorm(self):
        return np.array(self._gnorm)
    


#############################

def main(args):
    """ Main entry point of the app """
    print(args)
    D=100
    np.random.seed(args.seed)

    # Generate eigen spectrum for matrix to optimize
    rho=0.6
    em=0.5
    eM=100
    i=np.arange(1,D+1)
    E=em + (eM-em)/(D-1)*(np.power(rho,D-i)*(i-1))

    # E=generate_spectrum(values=((0.5,1),(1,100)),position=[0,0.85,1],N=100,frac=True)

    # print([f"{ev:.1e}" for ev in sorted(E)])

    P = random_rotation_matrix(D)
    A = (P*E).dot(P.T)

    testfun=byNumber(1,Q=A,xsol=1*np.random.randn(D)-2)
    x0 = 5*np.random.randn(D)

    g0 = testfun.g(x0)
    W = np.ones((D,1))*np.linalg.norm(g0)

    # convergence_options={'ftol':None,'gtol':1e-7}
    convergence_options={'ftol':None,'gtol':1e-5}


    # plt.style.use("icml21.mplstyle")


    ###################
    # GP-Hessian
    ###################
    options={'memory':None,**convergence_options}
    gph_runner=RunInformation(testfun,testfun.g,**options)
    gph_runner.save_data(x0)
    kern=kernels.FastLinearKernel(D,W)
    HessianMin(x0,testfun,testfun.g,kern,callback=gph_runner.save_data,**options)
    
    ###################
    # GP-Gradient
    ###################
    options={'memory':None,**convergence_options}
    gpg_runner=RunInformation(testfun,testfun.g,**options)
    gpg_runner.save_data(x0)
    kern=kernels.FastLinearKernel(D,W)

    GradientMin(x0,testfun,testfun.g,kern,callback=gpg_runner.save_data,**options)
    # GradientMinFixed(x0,testfun,testfun.g,kern,callback=gpg_runner.save_data,**options)


    ###################
    # CG
    ###################
    options={**convergence_options}
    cg_runner=RunInformation(testfun,testfun.g,**options)
    cg_runner.save_data(x0)
    CG(x0,testfun,testfun.g,callback=cg_runner.save_data,**options)
    
    # f0=cg_runner.fval[0]
    f0=1.0
    # Plotting
    plt.semilogy(cg_runner.fval/f0,color='C2',ls='-',label='CG')
    plt.semilogy(gph_runner.fval/f0,color='C0',ls=':',label='GP-H')
    plt.semilogy(gpg_runner.fval/f0,color='C1',ls='--',label='GP-X')
    plt.ylabel(r"$f(\mathbf{x})-f(\mathbf{x}_*)$")
    plt.xlabel(r"Iteration")
    # plt.title(r"Quadratic-100d")
    plt.legend()



    if args.save:
        savename=args.savedir+f"lin_{D}.pdf"
        print(f"Saving figure as: {savename}")
        plt.savefig(savename)
    else:
        
        plt.show()



if (__name__ == "__main__"):
    print('Executing as standalone script')
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # # Optional float
    # parser.add_argument("-lw", "--linewidth", metavar='L',
    #                     type=float, default=5.5)
    # Optional float
    parser.add_argument("--width", metavar='W', type=float, default=1.0)
    # Optional float
    parser.add_argument("--height", metavar='H',
                        type=float)
    # parser.add_argument("--style", metavar='S', default=None,
    #                     choices=['icml21', 'presentation'])
    parser.add_argument("--seed", metavar='s',
                        type=int,default=42)
    parser.add_argument("--savedir", metavar='loc', type=str ,default="../fig/",)

    # Optional argument flag which defaults to False
    parser.add_argument("-s", "--save", action="store_true", default=False)

    # Optional choices
    # parser.add_argument('-o','--optimizer', choices=['bfgs', 'dfp'],default='dfp')

    args = parser.parse_args()
    main(args)
