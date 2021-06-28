import numpy as np
import itertools


# base_command = "python approximate_inference.py"
base_commands = ["python approximate_inference.py"]

DIR = './'



tol = [f'{eta:.1e}' for eta in np.logspace(-6, -2, 5)]
# w = reversed([f'{eta:.1e}' for eta in np.logspace(-6, -2, 15)])
w = reversed([f'{eta:.1e}' for eta in np.logspace(-8, -6, 5)])


options = {
    '--D':[100],
    '--N':[1000],
    # '--lamda':[0.0,1e-3,1e-2],
    '--lamda':[0.0],
    '--seed':[42],
    '--tol': tol,
    '--w':w,
    # '--tol': [1e-8],
    # '--w':[1e-3],
    '--style': [0],
    '--maxiter':[100000]
}

flags = ['--save']


def make_all_combinations(options):
    keys, values = zip(*options.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return experiments


experiments = make_all_combinations(options)
# print(experiments)
print(
    f'Generating runfile for {len(experiments)} experiments with base_command:{base_commands} @ {DIR}')
with open(DIR + "jobs.txt", "w") as f:

    for base_command in base_commands:
        for exp in experiments:
            command = base_command
            for k, v in exp.items():
                command += f" {k}={v}"

                # if k == '--testproblem':
                #     command += f" {v}"
                # else:
                #     command += f" {k}={v}"

            for flag in flags:
                command+=f" {flag}"

            command += "\n"
            # print(command)
            f.write(command)
