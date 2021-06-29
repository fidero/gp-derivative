import numpy as np
from kernels import RBF, ExponentialKernel, LinearKernel, FastLinearKernel
from inference import DerivativeGaussianProcess
from matplotlib import pyplot as plt
from Fcode import plot_matrices
import scipy.linalg as sla

if __name__ == '__main__':
    seed = 42
    D=10
    N=3
    np.random.seed(seed)
    nugget = 1e-8

    c  = np.ones((D,1))
    # c  = np.ones((D,1))*0

    dX = np.random.randn(D,N) + 5 #/D
    V= np.random.randn(D,N)
    A=V.dot(V.T)+np.eye(D)
    dY = A @ (dX-c)  #np.random.randn(D,N)*D+ 5
    dY_=dY.reshape(-1,1,order='f')
    # c  = np.mean(dX,axis=1,keepdims=True)
    
    # w = np.ones((D,1))
    # w = np.pi * np.ones((D,1))
    w = 1/(np.ones((D,1))+np.pi*np.random.rand(D,1))

    print(f"  seed: {seed}\n     D: {D}\n     N: {N}\nnugget:{nugget:.1e}")
    

    for K in [RBF,ExponentialKernel,LinearKernel,FastLinearKernel]:

        print("===============\n",K)
        try:
            kern=K(D,w=w,c=c)    
        except TypeError as e:
            kern=K(D,w=w)
        

        dgp = DerivativeGaussianProcess(kern, nugget=nugget)
        dgp.condition(dX=dX,dY=dY)

        G = kern._dKd_explicit(dX,dX) + np.kron(nugget*np.eye(N),np.diag(w[:,0]))
        # Z = np.linalg.solve(G,dY_).reshape(*dX.shape,order='f')
        Z = sla.solve(G,dY_,assume_a='pos').reshape(*dX.shape,order='f')
        # Z = sla.solve(G,dY_,assume_a='sym').reshape(*dX.shape,order='f')
        print(f"|Z_explicit - Z_fast|: {np.linalg.norm(Z-dgp.Z)**2/(D*N)}")
        # print(np.linalg.eigvalsh(G))


    print("===============")