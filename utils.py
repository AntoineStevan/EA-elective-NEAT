from domain import GymTask, games

import numpy as np


def master():
    """Main NEAT optimization script
    """


def gatherData(data, neat, gen, hyp, savePop=False):
    """Collects run data, saves it to disk, and exports pickled population

    Args:
      data       - (DataGatherer)  - collected run data
      neat       - (Neat)          - neat algorithm container
        .pop     - [Ind]           - list of individuals in population
        .species - (Species)       - current species
      gen        - (ind)           - current generation
      hyp        - (dict)          - algorithm hyperparameters
      savePop    - (bool)          - save current population to disk?

    Return:
      data - (DataGatherer) - updated run data
    """
    data.gatherData(neat.pop, neat.species)
    if (gen % hyp['save_mod']) == 0:
        data = checkBest(data, hyp)
        data.save(gen)

    if savePop is True:  # Get a sample pop to play with in notebooks
        global fileName
        pref = 'log/' + fileName
        import pickle
        with open(pref + '_pop.obj', 'wb') as fp:
            pickle.dump(neat.pop, fp)

    return data


def checkBest(data, hyp):
    """Checks better performing individual if it performs over many trials.
    Test a new 'best' individual with many different seeds to see if it really
    outperforms the current best.

    Args:
      data - (DataGatherer) - collected run data

    Return:
      data - (DataGatherer) - collected run data with best individual updated


    * This is a bit hacky, but is only for data gathering, and not optimization
    """
    if data.newBest is True:
        # print('newBest True')
        bestReps = hyp['bestReps']
        task = GymTask(games[hyp['task']], nReps=hyp['alg_nReps'], budget=hyp["budget"])
        wVec = data.best[-1].wMat.flatten()
        aVec = data.best[-1].aVec.flatten()
        fitVector = []
        for i in range(bestReps):
            fitVector.append(task.getFitness(wVec, aVec))
        trueFit = np.mean(fitVector)
        # print("New evaluated true fitness : {:.2f} |***| Previous true fitness : {:.2f}".format(trueFit, data.best[-2].fitness))
        data.elite[-1].fitness = trueFit
        data.fit_max[-1] = trueFit

        if trueFit > data.best[-2].fitness:  # Actually better!
            data.best[-1].fitness = trueFit
            data.fit_top[-1] = trueFit
            data.bestFitVec = fitVector
            # print("Actually better!")
        else:  # Just lucky!
            prev = hyp['save_mod'] + 1
            data.best[-prev:] = data.best[-prev]
            data.fit_top[-prev:] = data.fit_top[-prev]
            data.newBest = False
            # print("Just lucky!")
    return data
