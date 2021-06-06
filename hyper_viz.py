import os

import matplotlib.pyplot as plt
import numpy as np

directory = "runs"

for run in os.listdir(directory):
    file = os.path.join(directory, run)
    arr = np.load(file)
    params = arr[1: int(arr[0])+1]
    fitnesses = arr[int(arr[0])+1:]
    plt.plot(fitnesses)
    plt.title(f"{run} - {params}")
    plt.show()
