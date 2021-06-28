#!/usr/bin/env python3
"""
Module Docstring
"""

# Imports
import argparse
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json
# Global variables

# Class declarations

# Function declarations
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

    # First apply appropriate styles
    if args.style is not None:
        for style in args.style:
            print("Applying style:",style)
            plt.style.use(style)

    # plt.style.use('icml21.mplstyle')


    # Fix relative scaling from stylefile
    figsize=[size*scale for size,scale in zip(mpl.rcParams['figure.figsize'],[args.width,args.height])]
    mpl.rcParams['figure.figsize']=figsize
    # mpl.rc
    # rm = ResultManager('./', problem='data')

    line_styles=[
        {'c':'C0','ls':'-','marker':'s'},
        {'c':'C1','ls':'-','marker':'X'},
        {'c':'C3','ls':'-','marker':'o'},
        {'c':'C2','ls':'--'},
        {'c':'0.4','ls':'-'},
        # {'c':'C4','ls':'-'},
        # {'c':'C5','ls':'-'},
        # {'c':'C6','ls':'-'},
        ]

    data = load_data_json(args.datadir)

    D_vals=set()



    for exp in data.values():
        D_vals.add(exp['D'])
    
    chol_data=defaultdict(list)
    dec_data=defaultdict(list)
    ratio_data=defaultdict(list)
    N_data=defaultdict(list)

    for exp in data.values():

        chol_data[exp['D']].append(np.mean(exp['cholesky_time']))
        dec_data[exp['D']].append(np.mean(exp['decomposition_time']))
        ratio_data[exp['D']].append(np.mean(np.asarray(exp['decomposition_time'])/np.asarray(exp['cholesky_time'])))
        N_data[exp['D']].append(exp['N'])


    # Exclude dimension 50 for clarity
    D_vals.remove(50)

    ###########################
    # Actual plotting goes here
    ###########################
    plt.figure()
    for i,D in enumerate(sorted(list(D_vals))):

        y=np.asarray([y for _,y in sorted(zip(N_data[D],ratio_data[D]))])
        x=np.asarray(sorted(N_data[D]))

        # y=[y for _,y in sorted(zip(N_data[D],ratio_data[D]))]
        # x=sorted(N_data[D])

        # plt.plot(x/D,y,**line_styles[i],label=f'D={D}')
        plt.semilogy(x/D,y,**line_styles[i],label=f'D={D}')
        # plt.semilogy(x/D,y,label=f'D={D}')

    plt.hlines(1.0,0,1.0,'k',ls='--')
    plt.xlabel('N/D')
    plt.ylabel('Woodbury/Cholesky')
    plt.legend()



    # Determine savename
    if args.save:
        filename=args.savename+'.pdf'
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

    parser.add_argument("--width", metavar='W', type=float, default=1.0)
    parser.add_argument("--height", metavar='H', type=float, default=1.0)

    parser.add_argument("--style", metavar='S', nargs="+", default=['icml21.mplstyle'])#,
                        # choices=['thesis','thesismargin','neurips20', 'presentation'])

    parser.add_argument('--figdir', type=str, default='../fig/',
                    help='Path to save location of figures')

    parser.add_argument('--datadir', type=str, default='../data/runtime/',
                    help='Path to data used for plotting')


    parser.add_argument('--savename', type=str, default='runtime',
                    help='Path to data used for plotting')

    # Optional argument flag which defaults to False
    parser.add_argument("-s", "--save", action="store_true", default=False)


    args = parser.parse_args()
    main(args)
