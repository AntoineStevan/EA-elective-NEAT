from matplotlib import pyplot as plt
import numpy as np
import argparse



def prettyPrint(data):
    x = np.linspace(1,len(data[0]),len(data[0]))
    y = np.mean(data, axis=0)
    print(y)
    std = np.std(data,axis=0)

    plt.plot(x,y,'k-',label='Mean')
    plt.xlabel("Generation")
    plt.ylabel("Max fitness")

    plt.fill_between(x, y-std, y+std, color='orange', label='Standard deviation', )
    plt.legend()
    plt.show()


if __name__ == "__main__":
## Parse input
    parser = argparse.ArgumentParser(description=('Pretty Print for Neat'))

    parser.add_argument('-d', '--directory', type=str, help='Directory Rewards', default='log/learn/')
    parser.add_argument('-f', '--file', type=str, help='Rewards', default='rewards.npy')

    args = parser.parse_args()
## End Parse Input

    prettyPrint(data=np.load(args.directory + args.file))




