import os

import matplotlib.pyplot as plt
import numpy as np

directory = "runs"

files = os.listdir(directory)
files.sort()

files = ["run_3.data.npy", "run_7.data.npy"]

for run in files:
    file = os.path.join(directory, run)
    arr = np.load(file)
    params = arr[1: int(arr[0])+1]
    fitnesses = arr[int(arr[0])+1:]
    print(params, fitnesses)
    plt.plot(fitnesses)
    plt.title(f"{run} - {params}")
    plt.show()
