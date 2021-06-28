from os import path
import pandas as pd


timestamp = '210204'
file = '../../out/hmc/banana_reps_' + timestamp

try:
    df = pd.read_csv(file, index_col=False)
except FileNotFoundError:
    print('{file} does not exist, run banana_reps.py or fix the date')

print(f"Mean HMC acceptance rate is {df['p_hmc'].mean()} +/- {df['p_hmc'].std()}\n"
      f"Mean GPG-HMC acceptance rate is {df['p_gp'].mean()} +/- {df['p_gp'].std()}\n\n"
      f"GP collects between {df['n_gp'].min()} and {df['n_gp'].max()} gradients"
      f"and takes {df['n_hmc'].mean()} +/- {df['n_hmc'].std()} HMC samples for training"
)
