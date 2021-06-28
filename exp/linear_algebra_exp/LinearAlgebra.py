import numpy as np
# import matplotlib.pyplot as plt
import sys
sys.path.append("../../src")

# import kernels
from inference import DerivativeGaussianProcess
# from utils import random_rotation_matrix, plot_matrices
# from test_function_optimization.functionsND import byNumber
# from nonlin_exp_fig import RunInformation



class ConvergenceChecker:
    
    def __init__(self,x0,g0,fun,ftol=None,gtol=None):
        self.g0=np.linalg.norm(g0)
        self.fun=fun
        self.f0=fun(x0)
        self.last_f=np.copy(self.f0)
        self.ftol=ftol
        self.gtol=gtol

    def converged(self,xi,gi):
        
        if (self.ftol is None) and (self.gtol is None):
            return False
        else:
        
            conditions_fulfilled=True
            if self.ftol is not None:
                fi=self.fun(xi)
                if fi/self.f0>self.ftol:
                    conditions_fulfilled=False
                self.last_f=fi

            if self.gtol is not None:
                if np.linalg.norm(gi)/self.g0>self.gtol:
                    conditions_fulfilled=False
            return conditions_fulfilled



def GradientMin(x0,fun,jac,kern,callback=None,ftol=None,gtol=None,maxiter=None,memory=None,disp=False):
    D=len(x0)
    dgp=DerivativeGaussianProcess(kern,nugget=0.0)
    
    xi=np.copy(np.asarray(x0))
    gi=jac(xi)
    X = np.copy(xi.reshape(-1,1))
    G = np.copy(gi.reshape(-1,1))
    dx=-gi
    
    gsol=np.zeros_like(x0).reshape(-1,1)
    
    CC=ConvergenceChecker(xi,gi,fun,ftol=ftol,gtol=gtol)
    if maxiter==None:
        maxiter=D  
    i=0
    while (i<maxiter) and not CC.converged(xi,gi) :
        i+=1
        Adx=jac(xi+dx)-gi
        a=-dx.T.dot(gi)/(dx.T.dot(Adx))
        
        xi += a*dx
        gi = jac(xi)

        if callback is not None:
            callback(xi)

        xi_ = xi.reshape(-1,1)
        gi_ = gi.reshape(-1,1)

        kern.update_hyperparameter("c",gi_)

        try:
            dgp.condition(dX=G,dY=X-xi_)
            dx = dgp.infer_g(gsol).ravel()
        except np.linalg.LinAlgError as e:
            # raise e
            break
        
        
        if memory is None:
            X=np.hstack((X,xi_))
            G=np.hstack((G,gi_))
        else:
            X=np.hstack((X,xi_))[:,-memory:]
            G=np.hstack((G,gi_))[:,-memory:]
        
        if disp:
            print(f"{i:3d}: f:{fun(xi)[0]:.1e} |g|:{np.linalg.norm(gi):.1e}")
        


def GradientMinFixed(x0,fun,jac,kern,callback=None,ftol=None,gtol=None,maxiter=None,memory=None,disp=False):
    D=len(x0)
    dgp=DerivativeGaussianProcess(kern,nugget=0.0)
    
    xi=np.copy(np.asarray(x0))
    gi=jac(xi)
    X = np.copy(xi.reshape(-1,1))
    G = np.copy(gi.reshape(-1,1))
    dx=-gi

    xnull=np.zeros_like(xi).reshape(-1,1)
    b=jac(xnull[:,0]).reshape(-1,1)
    kern.update_hyperparameter("c",b)

    gsol=np.zeros_like(x0).reshape(-1,1)
    
    CC=ConvergenceChecker(xi,gi,fun,ftol=ftol,gtol=gtol)
    if maxiter==None:
        maxiter=D  
    i=0
    while (i<maxiter) and not CC.converged(xi,gi) :
        i+=1
        Adx=jac(xi+dx)-gi
        a=-dx.T.dot(gi)/(dx.T.dot(Adx))
        
        xi += a*dx
        gi = jac(xi)

        if callback is not None:
            callback(xi)

        xi_ = xi.reshape(-1,1)
        gi_ = gi.reshape(-1,1)


        try:
            dgp.condition(dX=G,dY=X-xnull)
            dx = dgp.infer_g(gsol).ravel()
        except np.linalg.LinAlgError as e:
            # raise e
            break
        
        
        if memory is None:
            X=np.hstack((X,xi_))
            G=np.hstack((G,gi_))
        else:
            X=np.hstack((X,xi_))[:,-memory:]
            G=np.hstack((G,gi_))[:,-memory:]
        
        if disp:
            print(f"{i:3d}: f:{fun(xi)[0]:.1e} |g|:{np.linalg.norm(gi):.1e}")
        



def HessianMin(x0,fun,jac,kern,callback=None,ftol=None,gtol=None,maxiter=None,memory=None,disp=False):
    D=len(x0)
    dgp=DerivativeGaussianProcess(kern,nugget=0.0)
#     dgp=DerivativeGaussianProcess(kern,nugget=1e-6)

    xi=np.copy(np.asarray(x0))#.reshape(-1,1)
    
    xnull=np.zeros_like(xi).reshape(-1,1)
    b=jac(xnull[:,0]).reshape(-1,1)
    kern.update_hyperparameter("c",xnull)
    
    gi=jac(xi)

    dx=-gi
    
    X = np.copy(xi.reshape(-1,1))
    G = np.copy(gi.reshape(-1,1))


    
    CC=ConvergenceChecker(xi,gi,fun,ftol=ftol,gtol=gtol)
    if maxiter==None:
        maxiter=D  
    i=0
    while (i<maxiter) and not CC.converged(xi,gi) :
        Adx=jac(xi+dx)-gi
        a=-dx.T.dot(gi)/(dx.T.dot(Adx))
        

        xi += a*dx
        gi = jac(xi)
        
        xi_=xi.reshape(-1,1)
        gi_=gi.reshape(-1,1)
        if memory is None:
            X=np.hstack((X,xi_))
            G=np.hstack((G,gi_))
        else:
            X=np.hstack((X,xi_))[:,-memory:]
            G=np.hstack((G,gi_))[:,-memory:]
        
        
        
        if callback is not None:
            callback(xi)

        try:
            dgp.condition(dX=X,dY=G-b)
            H=dgp.infer_h(xi_)
        
            dx = -H.invmul(gi_).ravel()
        except np.linalg.LinAlgError as e:
            break
        
        
        
        
        i+=1
        
        
        
        if disp:
            print(f"{i:3d}: f:{fun(xi)[0]:.1e} |g|:{np.linalg.norm(gi):.1e}")



def CG(x0,fun,jac,callback=None,ftol=None,gtol=None,maxiter=None,disp=False):
    D=len(x0)
    xi=np.copy(np.asarray(x0))
    ri=-jac(x0)

    p=ri

    
    CC=ConvergenceChecker(xi,-ri,fun,ftol=ftol,gtol=gtol)
    if maxiter==None:
        maxiter=D  
    i=0
    
    while (i<maxiter) and not CC.converged(xi,-ri):
        i+=1
        Ap=jac(xi+p)+ri
        
        a = (p.T @ ri/(p.T @ Ap)).ravel()
        
        
        xi+=a*p
        
        ri_ =-jac(xi)
        

#         b = np.dot(ri_.T,Ap)/np.dot(p.T,Ap)

        if callback is not None:
            callback(xi)
        

        b = (np.dot(ri_.T,ri_)/np.dot(ri.T,ri)).ravel()
        
        p=ri_+b*p
        ri=ri_

        if disp:
            print(f"{i:3d}: f:{fun(xi)[0]:.1e} |g|:{np.linalg.norm(ri):.1e}")