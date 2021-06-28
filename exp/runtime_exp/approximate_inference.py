import sys

sys.path.append('../')

from test_functions.functionsND import byNumber
import numpy as np 
# from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import cg, LinearOperator
import argparse
import json 
import time


class RBF:
    def __init__(self,th=1.0,w=1.0,lam=1e-1):
        self.th=th
        self.w=w
        self.lam=lam


    def build(self,X):
        D,N=X.shape
        self.D=D
        self.N=N
        self.WX=(X.T*self.w).T
        self.Kpp = self.K(X,X)
        self.Kp = self.K(X,X) + self.lam/self.w*np.eye(N)


    def K(self, xa, xb):
        W = self.w
        XTWX = xa.T.dot(W * xb)
        D = (
            np.sum(xa * W * xa, axis=0, keepdims=True).T
            + np.sum(xb * W * xb, axis=0, keepdims=True)
            - 2 * XTWX
        )
        return self.th*np.exp(-0.5 * D)

    def dKd_explicit(self, xa, xb):
        """ Explicitly built kernel derivative w.r.t. first and second argument in for loops"""
        Ma = xa.shape[1]
        Mb = xb.shape[1]
        N  = xa.shape[0]
        Kp = self.K(xa, xb) +self.lam/self.w*np.eye(Ma)
        Kpp = self.K(xa, xb)

        W = np.diag(self.w*np.ones(N))

        Wxa = W @ xa
        Wxb = W @ xb

        G = np.zeros((N * Ma, N * Mb))

        for a in range(Ma):
            for b in range(Mb):
                Wd = Wxa[:, a] - Wxb[:, b]
                G[a * N : (a + 1) * N, b * N : (b + 1) * N] = Kp[a, b] * W - Kpp[a, b] * np.outer(Wd, Wd)
                # G[a * N : (a + 1) * N, b * N : (b + 1) * N] =  - Kpp[a, b] * np.outer(Wd, Wd)

        return G

    def matvec(self,v):
        V=v.reshape(self.D,self.N,order='f')
        M=V.T.dot(self.WX)
        M=self.Kpp*(M-np.diag(M).reshape(-1,1))
        d = np.sum(M,axis=0)
        M[np.arange(self.N),np.arange(self.N)]=-d

        V=self.w*V@self.Kp + self.WX @ M
        # return V.reshape(-1,1,order='f')
        return V.reshape(*v.shape,order='f')
        # return V.reshape(-1,order='f')



class RunInformation:
    
    def __init__(self,G,b,x0,**params):
        self.b=b
        self.G=G
        # self.g0=np.linalg.norm(b-G(x0))

        self._gnorm=[np.linalg.norm(b-G(x0))]
        self._params={**params}
        self.i=0
        
    def save_data(self,xk):
        # self._gnorm.append(np.linalg.norm(self.b-self.G(xk)))
        self._gnorm.append(self._gnorm[0])

        self.i+=1

    @property
    def gnorm(self):
        return np.array(self._gnorm)
    
def genX(D,N,seed=42,style=0):
    np.random.seed(seed)
    if style==0:
        return 4*np.random.rand(D,N)-2
    elif style==1:
        return 2*np.random.randint(low=-1,high=1,size=(D,N))
    elif style==2:
        return np.random.randint(low=-2,high=2,size=(D,N))

    else:
        raise ValueError(f"style not recognized")


def main(args):

    D=args.D
    N=args.N
    tol=args.tol
    w=args.w
    lam=args.lamda
    th =args.theta
    seed =args.seed
    maxiter=args.maxiter
    style=args.style

    fun=byNumber(6,N=D,a=0,b=2.0)

    # print(f"(N,D)=({N},{D})| seed:{seed} tol:{tol:.1e} w:{w:.1e} lam:{lam:.1e} th:{th:.1e}")
    
    # np.random.seed(seed)
    X=genX(D,N,seed,style)
    g = fun.g(X)
    # g=np.abs(np.random.randn(*X.shape))*X
    g_=g.reshape(-1,order='f')

    kern=RBF(lam=lam,w=w,th=th)
    kern.build(X)

    x0=np.copy(g_)
    # x0=np.zeros_like(g_)

    G = LinearOperator(shape=(N*D,N*D),matvec=kern.matvec,rmatvec=kern.matvec)

    RI=RunInformation(G=G,b=g_,x0=x0)
    # G2=kern.dKd_explicit(X,X)
    # print(np.allclose(G(v),G2 @ v))

    start=time.time()
    alpha,info=cg(A=G,b=g_,x0=x0,tol=tol,maxiter=maxiter,callback=RI.save_data)
    elapsed=time.time()-start
    RI._gnorm.append(np.linalg.norm(G(alpha)-g_))
    save_data={
    'R':args.style,
    'w':args.w,
    'tol':args.tol,
    'D':args.D,
    'N':args.N,
    'lamda':args.lamda,
    'theta':args.theta,
    'seed':args.seed,
    'maxiter':args.maxiter,
    }

    savename=""
    for k,v in save_data.items():
        savename+=f"{v}_"
    savename+=".json"
    
    # file_name = f"{info['optimizer_name']}_lr_{info['lr']:.4f}_{info['timestamp']}_xxx{info['identifier']}"
    
    save_data={**save_data,'iterations':RI.i,
    'gnorm':RI._gnorm,
    'alpha':alpha.reshape(D,N,order='f').tolist(),'succes':info,'elapsed':elapsed}
   
    print(f"({D},{N}) {RI.i}: {tol} {w}  {elapsed:.2f} {info}")
    if args.save:
        savename =args.root+savename
        with open(savename, 'w') as fp:
            print('Saving @: ', savename)
            json.dump(save_data, fp, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient inference')
    parser.add_argument('--N', type=int, default=10, metavar='N',
                        help='# observations')
    parser.add_argument('--D', type=int, default=10, metavar='D',
                        help='# dimensions')
    parser.add_argument('--w', type=float, default=1.0, metavar='W',
                        help='w-scaling rbf')
    parser.add_argument('--theta', type=float, default=1.0, metavar='TH',
                        help='theta-scaling rbf')
    parser.add_argument('--lamda', type=float, default=0.0, metavar='L',
                        help='noise')
    parser.add_argument('--maxiter', type=int, default=None, metavar='I',
                        help='maximum iterations')
    parser.add_argument('--tol', type=float, default=1e-5, metavar='T',
                        help='maximum iterations')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--style', type=int, default=1, metavar='Rand',
                        help='0')
    parser.add_argument('--root', type=str, default='data/', metavar='R',
                        help='relative save directory (default: "")')

    parser.add_argument('--save', action='store_true', default=False,
                        help='For Saving the current Model')
    # parser.add_argument('--zero', action='store_true', default=False,
    #                     help='start from 0')

    cli_args,unknown = parser.parse_known_args()
    # print(cli_args)
    # print(type(cli_args))
    main(cli_args)




    
    

