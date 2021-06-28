#!/usr/bin/env python3



# Imports
import argparse
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
sys.path+=["../../src",'../']

from Optimization import GPHessianMinimizer, GPGradientMinimizer

import kernels
from test_function_optimization.functionsND import byNumber


# Class declarations
class RunInformation:
    '''Convenience class for storing required information for plotting.
    '''
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


#############################

def main(args):
    """ Main entry point of the app """
    print(args)
    D=100
    testfun=byNumber(6,N=D,a=0,b=2.0)
    x0=testfun.x0.reshape(-1,1)
    gtol=1e-2
    base_opts={'gtol':gtol}

    # plt.style.use("icml21.mplstyle")

    ###################
    # GP-Hessian
    ###################
    # w=10e0*np.ones((D,1))
    w=9e0*np.ones((D,1))
    kern=kernels.RBF(D,w)
    options={'kern':kern,'memory':2,'nugget':1e-6,**base_opts}
    gph_runner=RunInformation(testfun.f,testfun.g,**options)
    gph_runner.save_data(x0)
    resGPH=minimize(fun=testfun.f,x0=x0,jac=testfun.g,method=GPHessianMinimizer,callback=gph_runner.save_data,options=options)

    ###################
    # GP-gradient
    ###################
    # w=3e-2*np.ones((D,1))
    w=5e-2*np.ones((D,1))
    kern=kernels.RBF(D,w)
    options={'kern':kern,'memory':2,'nugget':1e-6,'fixed_w':True,**base_opts}
    gpg_runner=RunInformation(testfun.f,testfun.g,**options)
    gpg_runner.save_data(x0)
    resGPG=minimize(fun=testfun.f,x0=x0,jac=testfun.g,method=GPGradientMinimizer,callback=gpg_runner.save_data,options=options)




    ###################
    # BFGS
    ###################
    options={**base_opts}
    bfgs_runner=RunInformation(testfun.f,testfun.g,**options)
    bfgs_runner.save_data(x0)
    resBFGS=minimize(fun=testfun.f,x0=x0,jac=testfun.g,callback=bfgs_runner.save_data,options=options)


    # Plotting
    plt.semilogy(bfgs_runner._fval,color='C3',ls='-',label='BFGS')
    plt.semilogy(gph_runner._fval,color='C0',ls=':',label='GP-H')
    plt.semilogy(gpg_runner._fval,color='C1',ls='--',label='GP-X')
    plt.ylim([1e-6,2e2])
    plt.yticks(np.power(10,2*np.arange(5))/10**6)
    plt.ylabel(r"$f(\mathbf{x})-f(\mathbf{x}_*)$")
    plt.xlabel(r"Iteration")
    # plt.title(r"Nonlinear-100d")
    plt.legend()

    if args.save:
        savename=args.savedir+"non_lin_100.pdf"
        print(f"Saving figure as: {savename}")
        plt.savefig(savename)
    else:
        
        plt.show()



if (__name__ == "__main__"):
    print('Executing as standalone script')
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()


    # Optional float
    parser.add_argument("--width", metavar='W', type=float, default=1.0)
    # Optional float
    parser.add_argument("--height", metavar='H',
                        type=float)
    # parser.add_argument("--style", metavar='S', default=None,
    #                     choices=['icml21', 'presentation'])

    parser.add_argument("--savedir", metavar='loc', type=str ,default="../fig/",)

    # Optional argument flag which defaults to False
    parser.add_argument("-s", "--save", action="store_true", default=False)

    # Optional choices
    # parser.add_argument('-o','--optimizer', choices=['bfgs', 'dfp'],default='dfp')

    args = parser.parse_args()
    main(args)
