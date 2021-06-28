import numpy as np
import itertools

# python runtime.py --N 30 --D 100 --w 0.001 --lamda 1.0 --repetitions 5 --root ./data/ --save

# base_command = "python approximate_inference.py"
base_commands = ["python runtime.py"]

DIR = './'



# tol = [f'{eta:.1e}' for eta in np.logspace(-6, -2, 5)]
# w = reversed([f'{eta:.1e}' for eta in np.logspace(-6, -2, 15)])

D=100
# Ns=[int(eta) for eta in np.linspace(0.1, 0.5, 5)*D]

Ns=[1,5]+[int(eta) for eta in np.linspace(0.1, 1.0, 10)*D]

variable_options=[
{'--D':[25],'--N':[1,2,5]+[int(eta) for eta in np.linspace(0.1, 1.0, 10)*25],'--w':[1e-3]},
{'--D':[50],'--N':[1,2,5]+[int(eta) for eta in np.linspace(0.1, 1.0, 10)*50],'--w':[1e-3]},
{'--D':[100],'--N':[1,2,5]+[int(eta) for eta in np.linspace(0.1, 1.0, 10)*100],'--w':[1e-3]},
{'--D':[250],'--N':[1,2,5]+[int(eta) for eta in np.linspace(0.1, 1.0, 10)*250],'--w':[1e-3]},
{'--D':[500],'--N':[1,2]+[int(eta) for eta in np.linspace(0.01, 0.2, 10)*500],'--w':[1e-3]},
{'--D':[1000],'--N':[1,5]+[int(eta) for eta in np.linspace(0.001, 0.04, 10)*1000],'--w':[1e-3]},
]


fixed_options={
    '--lamda':[1.0],
    # '--w':[0.01],
    '--seed':[42],
    '--repetitions':[3],
    '--root': ['./data/'],
    # '--tol': [1e-8],
    }

# options = {
#     '--D':[D],
#     '--N':Ns,
#     # '--lamda':[0.0,1e-3,1e-2],
# }

flags = ['--save']


def make_all_combinations(options):
    keys, values = zip(*options.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return experiments

with open(DIR + "jobs.txt", "w") as f:
    f.write("")

for vo in variable_options:
    
    options={**vo,**fixed_options}
    # print(options)
    experiments = make_all_combinations(options)
    # print(experiments)
    print(
        f'Generating runfile for {len(experiments)} experiments with base_command:{base_commands} @ {DIR}')
    with open(DIR + "jobs.txt", "a") as f:

        for base_command in base_commands:
            for exp in experiments:
                
                command = base_command
                for k, v in exp.items():
                    command += f" {k}={v}"

                for flag in flags:
                    command+=f" {flag}"

                command += "\n"
                f.write(command)
