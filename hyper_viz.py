import os

import matplotlib.pyplot as plt
import numpy as np

directory = "runs"
show = False
Bestfit =-1
BestParams = []
iteration = 0

for run in os.listdir(directory):
    iteration+=1
    file = os.path.join(directory, run)
    arr = np.load(file)
    params = arr[1: int(arr[0])+1]
    fitnesses = arr[int(arr[0])+1:]
    if np.amax(fitnesses) > Bestfit:
        Bestfit = np.amax(fitnesses)
        BestParams = params
        BestIteration = iteration
    if show:
        plt.plot(fitnesses)
        plt.draw()
        plt.pause(1)
        plt.clf()

np. set_printoptions(suppress=True)
print("Meilleure fitness au set {} : {}".format(BestIteration, Bestfit))
print(BestParams)
