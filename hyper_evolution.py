import argparse
from collections import OrderedDict
from itertools import product
from pprint import pprint

import numpy as np

from utils import gatherData

np.set_printoptions(precision=2, linewidth=160)

from prettyNEAT.dataGatherer import DataGatherer  # prettyNeat
from prettyNEAT.neat import Neat
from domain import (
    loadHyp,
    updateHyp,
    GymTask,
    games
)  # Task environments


# -- Run NEAT ------------------------------------------------------------ -- #

def run_one_hyp(hyp, run_params, nb):
    for param in run_params:
        hyp[param] = run_params[param]

    data = DataGatherer(fileName, hyp)
    neat = Neat(hyp)

    task = GymTask(games[hyp['task']], nReps=hyp['alg_nReps'], budget=hyp["budget"])
    gens = range(hyp['maxGen'])
    for gen in gens:
        pop = neat.ask()  # Get newly evolved individuals from NEAT
        reward = np.empty(len(pop), dtype=np.float64)
        for i in range(len(pop)):
            wVec = pop[i].wMat.flatten()
            aVec = pop[i].aVec.flatten()
            reward[i] = task.getFitness(wVec, aVec)  # process it
            if task.curr_eval >= task.budget:
                break
        neat.tell(reward)  # Send fitness to NEAT

        if task.curr_eval >= task.budget:
            break
        data = gatherData(data, neat, gen, hyp)
        print(f"{gen}\t{data.display()}, |---| budget: {task.curr_eval} / {task.budget}", end='\r')

    # Clean up and data gathering at run end
    # data = gatherData(data, neat, gen, hyp, savePop=True)
    values = list(run_params.values())
    np.save(f"runs/run_{nb}.data", [len(values)] + values + list(data.fit_top))

    return data.fit_top[-1]


class RunBuilder:
    @staticmethod
    def get_runs(parameters):
        runs = []
        for v in product(*parameters.values()):
            runs.append(dict(zip(parameters.keys(), v)))

        return runs


if __name__ == "__main__":
    ''' Parse input and launch '''
    parser = argparse.ArgumentParser(description=('Evolve NEAT networks'))

    parser.add_argument('-d', '--default', type=str, \
                        help='default hyperparameter file', default='config/default_neat.json')

    parser.add_argument('-p', '--hyperparam', type=str, \
                        help='hyperparameter file', default=None)

    parser.add_argument('-o', '--outPrefix', type=str, \
                        help='file name for result output', default='test')

    parser.add_argument('-n', '--num_worker', type=int, \
                        help='number of cores to use', default=8)

    args = parser.parse_args()
    # +== EA-elective-NEAT =============================================================================================
    pprint(args.__dict__)
    # ==================================================================================================================
    fileName = args.outPrefix
    hyp_default = args.default
    hyp_adjust = args.hyperparam

    hyp = loadHyp(pFileName=hyp_default, printHyp=True)
    updateHyp(hyp, hyp_adjust)
    parameters = OrderedDict(
        popSize=[64, 200],

        prob_addConn=[.025, .1],
        prob_addNode=[.015, .06],
        prob_crossover=[.7, .9],
        prob_enable=[.005, .02],
        prob_mutConn=[.7, .9],
        prob_initEnable=[.8, 1.],
    )

    print(list(parameters.keys()))
    runs = RunBuilder.get_runs(parameters)
    t = range(60, 100)
    b_fit = 0
    b_run = -1
    for run in t:
        fitness = run_one_hyp(hyp, runs[run], run)
        if fitness > b_fit:
            b_fit = fitness
            b_run = run
        comment = f"run - {run} - fitness: {fitness} |---| b_fit - {b_fit} ({b_run}) |---| params - {list(runs[run].values())}\t\t\t\t{run} / {len(runs)}"
        print(comment)
