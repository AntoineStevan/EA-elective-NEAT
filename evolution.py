import argparse
import time
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

    # +== EA-elective-NEAT =============================================================================================
    print("data will be stored in", fileName)
    # ==================================================================================================================
    data = DataGatherer(fileName, hyp)
    neat = Neat(hyp)

    t_start = time.time()
    task = GymTask(games[hyp['task']], nReps=hyp['alg_nReps'], budget=hyp["budget"])
    for gen in range(hyp['maxGen']):
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
        
        t = time.time() - t_start
        prev_t = hyp["budget"] / task.curr_eval * t
        print(gen, '\t - \t', data.display(), f"|---| budget: {task.curr_eval} / {task.budget}", end=' ')
        tt = int(t)
        pt = int(prev_t)
        print(f"\t|---| {tt // 60}:{tt % 60} / {pt // 60}:{pt % 60}")

    # Clean up and data gathering at run end
    data = gatherData(data, neat, gen, hyp, savePop=True)
    data.save()
    data.savePop(neat.pop, fileName)  # Save population as 2D numpy arrays
