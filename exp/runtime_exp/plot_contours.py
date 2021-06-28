#!/usr/bin/env python3
"""
Script to run and 
If fonts are not loading execute:
rm -r ~/.cache/matplotlib
"""


# Imports
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
sys.path.append("../src")
sys.path.append("./cloud/")


import kernels
from test_function_optimization.functionsND import byNumber
import json
from approximate_inference import genX
# Global variables

# Class declarations

def load_data_json(path):
    assert os.path.exists(path), f"{path} not found"

    stored_data = {}

    for root, dirs, files in os.walk(path):
        for file in files:

            filename, ending = os.path.splitext(file)

            if ending == '.json':
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)

                basename = os.path.split(root)[-1] +'_'+filename
                keyname = basename
                i = 0
                while keyname in stored_data.keys():
                    i += 1
                    keyname = basename+f'_{i}'

                stored_data[keyname] = {**data}

    return stored_data


#############################


def main(args):
    """ Main entry point of the app """
    print(args)
    DATADIR=args.datadir
    FILE="chosen.json"
    # FILE="chosen2.json"
    # FILE="0_0.001_1e-06_100_1000_0.0_1.0_42_50000_.json"

    # cmap = "cividis"
    plt.style.use("icml21.mplstyle")
    ticks=[-2,-1,0,1,2]
    figsize=(3.25,1.625)
    # figsize=(3.25,1.625)
    x_lim=[-2,2]
    y_lim=[-2,2]
    ngrid=100


    # Read parameters from file
    with open(os.path.join(DATADIR, FILE), 'r') as f:
        exp = json.load(f)

    for k,v in exp.items():
        if k not in {'alpha','gnorm'}:
            print(f"{k}:\t {v}")


    print(f"{exp['iterations']} {exp['tol']} {exp['w']} {exp['elapsed']}")
    D=exp['D']
    N=exp['N']

    x_plot=np.linspace(*x_lim,ngrid)
    y_plot=np.linspace(*y_lim,int(ngrid*np.diff(y_lim)[0]/np.diff(x_lim)[0]))
    X1, X2 = np.meshgrid(x_plot, y_plot, indexing='ij')
    # X = 0*np.ones((D,*X1.shape))
    X = np.zeros((D,*X1.shape))

    # Chosed 2 consecutive dimensions
    i=0
    X[i:i+2]=np.stack((X1,X2),axis=0)

    X_=X.reshape(D,-1,order='f')


    # Predict
    Xgp=genX(D,N,exp['seed'])
    Z=np.array(exp['alpha'])
    kern=kernels.RBF(input_dim=D,w=exp['w'])

    Fpred = kern._Kd_mv(X_,Xgp,Z).reshape(ngrid,ngrid,order='f')
    Fpred-=np.min(Fpred)


    #Ground truth
    fun=byNumber(6,N=D,a=0,b=2.0)
    Fgt=fun(X_).reshape(ngrid,ngrid,order='f')
    Fgt-=np.min(Fgt)


    # Plotting

    fig,axarr=plt.subplots(1,2,figsize=figsize,constrained_layout=True)

    for ax,title,F in zip(axarr,["Ground Truth","RBF"],[Fgt,Fpred]):
        ax.contourf(X1, X2, np.log10(1+F))
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        




    if args.save:
        savename=args.savedir+f"contour_100_{i}.pdf"
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
    parser.add_argument("--style", metavar='S', default=None,
                        choices=['icml21', 'presentation'])

    parser.add_argument("--savedir", metavar='loc', type=str ,default="../fig/")
    parser.add_argument("--datadir", metavar='dloc', type=str ,default="../data/")

    # Optional argument flag which defaults to False
    parser.add_argument("-s", "--save", action="store_true", default=False)

    # parser.add_argument("-s", "--truth", action="store_true", default=False)

    # Optional choices
    # parser.add_argument('-o','--optimizer', choices=['bfgs', 'dfp'],default='dfp')

    args = parser.parse_args()
    main(args)
