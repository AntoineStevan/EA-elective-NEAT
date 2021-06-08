from prettyNEAT.ann import *
import networkx as nx
import matplotlib.pyplot as plt
import argparse


def printNN(file, show=False):
    ind = np.loadtxt(file, delimiter=',')
    wMat = ind[:, :-1]  # Weight Matrix

    neural_graph = nx.Graph()

    input = 0
    output = 0
    hidden = 0
    dead = 0
    color_map = []

    # Generation de tous les noeuds du graph
    for i in range(len(wMat)):
        if np.all(wMat[i, :] == 0) and np.all(wMat[:, i] == 0):
            dead += 1
            neural_graph.add_node(i)
            color_map.append('#d0d241')
        elif np.all(wMat[i, :] == 0):
            input += 1
            neural_graph.add_node(i)
            color_map.append('blue')

        elif np.all(wMat[:, i] == 0):
            output += 1
            neural_graph.add_node(i)
            color_map.append('red')

        else:
            hidden += 1
            neural_graph.add_node(i)
            color_map.append('#959f91')

    # Generation des liaisons
    for i in range(len(wMat)):
        for j in range(len(wMat[:, i])):

            if not wMat[j, i] == 0:
                neural_graph.add_edge(i, j, weight=wMat[j, i])

    nx.draw_networkx(neural_graph, node_color=color_map, with_labels=False)
    plt.show()

    if show:
        print("input : " + str(input))
        print("hidden : " + str(hidden))
        print("output : " + str(output))
        print("non connecté : " + str(dead))
    return input, hidden, output, dead


if __name__ == "__main__":
    ## Parse input
    parser = argparse.ArgumentParser(description=('Pretty Print for Neat'))

    parser.add_argument('-d', '--directory', type=str, help='Directory Rewards', default='log/freeway/')
    parser.add_argument('-f', '--file', type=str, help='Rewards', default='res_best.out')

    args = parser.parse_args()

    ## End Parse Input
    input, hidden, output, dead = printNN(args.directory + args.file, show=True)
