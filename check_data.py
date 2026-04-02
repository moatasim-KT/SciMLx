import numpy as np
from prepare import _generate_dataset
import os

benchmark = "darcy_2d"
n = 10
seed = 42

inputs, targets = _generate_dataset(benchmark, n, seed)
print(f"Inputs NaN: {np.isnan(inputs).any()}")
print(f"Targets NaN: {np.isnan(targets).any()}")
print(f"Targets Max: {np.max(np.abs(targets))}")
print(f"Targets Min: {np.min(np.abs(targets))}")
