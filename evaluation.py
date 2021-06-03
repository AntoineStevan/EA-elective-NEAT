import argparse

# from domain import *  # Task environments
from pprint import pprint

import numpy as np

from domain import (
    loadHyp,
    updateHyp,
    GymTask,
    games)
from prettyNEAT import (
    importNet,
    lsave)

np.set_printoptions(precision=2)
np.set_printoptions(linewidth=160)


def main(argv):
    """Tests network on task N times and returns mean fitness.
    """
    infile = argv.infile
    outPref = argv.outPref
    hyp_default = argv.default
    hyp_adjust = argv.hyperparam
    nRep = argv.nReps
    view = argv.view

    # Load task and parameters
    hyp = loadHyp(pFileName=hyp_default)
    updateHyp(hyp, hyp_adjust)
    task = GymTask(games[hyp['task']], nReps=hyp['alg_nReps'])

    # Bullet needs some extra help getting started
    if hyp['task'].startswith("bullet"):
        task.env.render("human")

    # Import and Test network
    wVec, aVec, wKey = importNet(infile)
    fitness = np.empty(1)
    fitness[:] = task.getFitness(wVec, aVec, view=view, nRep=nRep)

    print("[***]\tFitness:", fitness)
    lsave(outPref + 'fitDist.out', fitness)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    ''' Parse input and launch '''
    parser = argparse.ArgumentParser(description=('Test ANNs on Task'))

    parser.add_argument('-i', '--infile', type=str,
                        help='file name for genome input', default='log/test_best.out')

    parser.add_argument('-o', '--outPref', type=str,
                        help='file name prefix for result input', default='log/result_')

    parser.add_argument('-d', '--default', type=str,
                        help='default hyperparameter file', default='config/default_neat.json')

    parser.add_argument('-p', '--hyperparam', type=str,
                        help='hyperparameter file', default=None)

    parser.add_argument('-r', '--nReps', type=int,
                        help='Number of repetitions', default=1)

    parser.add_argument('-v', '--view', type=str2bool,
                        help='Visualize (True) or Save (False)', default=True)

    args = parser.parse_args()

    # +== EA-elective-NEAT =============================================================================================
    pprint(args.__dict__)
    # ==================================================================================================================

    main(args)
