# main.py (또는 cli.py)
from simulation import run_simulation_single
import numpy as np

params = {
    "p_list": [100, 200, 500, 1000, 2000],
    "a_list": [0.0, 0.5, 1/np.sqrt(2), np.sqrt(3)/2, 1.0],
    "n": 40,
    "nu": 5,
    "n_trials": 100,
    "sigma_coef": (1.0, 40.0),
    "master_seed": 725,
    "orthonormal_arg": True
}

run_simulation_single(params)