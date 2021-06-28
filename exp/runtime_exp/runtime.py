import sys
sys.path+=["../../src/","../"]
from test_functions.functionsND import byNumber

import kernels
from inference import DerivativeGaussianProcess
import numpy as np 
import scipy.linalg as sla
import argparse
import json 
import time

from calculate_storage import storageDecomposition,storageFull




    
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
    w=args.w
    lam=args.lamda
    seed =args.seed
    reps=args.repetitions
    style=args.style

    fun=byNumber(6,N=D,a=0,b=2.0)
    np.random.seed(seed)
    seeds=np.random.randint(0,100000,size=reps+1)
    # print(seeds)
    # print(f"(N,D)=({N},{D})| seed:{seed} tol:{tol:.1e} w:{w:.1e} lam:{lam:.1e} th:{th:.1e}")
    


    kern=kernels.RBF(D,w=w*np.ones((D,1)))
    
    cholesky_solve_time=[]
    decomposition_solve_time=[]


    
    # Decomposition solve
    for seed in seeds:
        np.random.seed(seed)
        X=genX(D,N,seed,style)
        g = fun.g(X)
        g_=g.reshape(-1,order='f')

        dgp=DerivativeGaussianProcess(kern,noise_variance=lam)

        start=time.process_time()
        dgp.condition(dX=X,dY=g)
        elapsed=time.process_time()-start
        # start=time.time()
        # dgp.condition(dX=X,dY=g)
        # elapsed=time.time()-start
        decomposition_solve_time.append(elapsed)

    # Cholesky solve
    for seed in seeds:
        np.random.seed(seed)
        X=genX(D,N,seed,style)
        g = fun.g(X)
        g_=g.reshape(-1,order='f')

        start=time.process_time()
        G = kern._dKd_explicit(X,X) + lam*w*np.eye(D*N)
        Z = sla.solve(G,g_, assume_a="pos")
        elapsed=time.process_time()-start

        # start=time.time()
        # G = kern._dKd_explicit(X,X) + lam*w*np.eye(D*N)
        # Z = sla.solve(G,g_, assume_a="pos")
        # elapsed=time.time()-start
        cholesky_solve_time.append(elapsed)

    G=[]
    
    save_data={
    'D':args.D,
    'N':args.N,
    'repetitions':args.repetitions,
    'w':args.w,
    'lamda':args.lamda,
    'R':args.style,
    'seed':args.seed,
    }

    savename=""
    for k,v in save_data.items():
        savename+=f"{v}_"
    savename+=".json"

    print(savename)
    
    
    save_data={**save_data,'cholesky_time':cholesky_solve_time,
    'decomposition_time':decomposition_solve_time,
    'decomposition_memory':storageDecomposition(N=N,D=D),
    'cholesky_memory':storageFull(N=N,D=D)}
    # print(cholesky_solve_time)
    # print(decomposition_solve_time)
    print(f"({D},{N}) {lam}: {w}  {np.asarray(decomposition_solve_time[1:])/np.asarray(cholesky_solve_time[1:])}")
    if args.save:
        savename =args.root+savename
        # savename =args.root+f"{D:d}_{}"
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
    parser.add_argument('--lamda', type=float, default=0.0, metavar='L',
                        help='noise')
    parser.add_argument('--repetitions', type=int, default=2, metavar='r',
                        help='Number of times to repeat experiment')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--style', type=int, default=0, metavar='Rand',
                        help='0')
    parser.add_argument('--root', type=str, default='data/', metavar='R',
                        help='relative save directory (default: "data/")')
    parser.add_argument("-s", "--save", action="store_true", default=False)

    cli_args,unknown = parser.parse_known_args()
    # print(cli_args)
    # print(type(cli_args))
    main(cli_args)




    
    

