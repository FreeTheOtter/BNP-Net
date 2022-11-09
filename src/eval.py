import numpy as np
import igraph as ig
from irm_undirected import irm
from irm_directed import irm_directed

def retrieve_samples(Z, gap = 25, burn_in = True):
    if burn_in == True:
        burn_in = len(Z)//2   
    return Z[burn_in::gap]

def cluster_summs(Z, ret = False):
    temp_Z = []
    mean_lenght = 0
    mean_nodes = 0
    for i in range(len(Z)):
        current_z = np.sum(Z[i], 0)
        temp_Z.append(current_z)
        mean_lenght += len(current_z)
        mean_nodes += np.mean(current_z)

    mean_lenght /= len(Z)
    mean_nodes /= len(Z)

    print('mean number of clusters', mean_lenght)
    print('mean nodes per cluster', mean_nodes)
    if ret:
        return temp_Z

g = ig.Graph.Read_GML('karate.txt')
X = np.array(g.get_adjacency().data)

T = 500
a = 1
b = 1
A = 5
Z = irm(X, T, a, b, A)

sample = retrieve_samples(Z)
cluster_summs(sample)

W = np.zeros((len(X),len(X)))
W[0,1] = 1

X_missing = X-X*W
Z_missing = irm(X_missing, T, a, b, A)
sample = retrieve_samples(Z_missing)
cluster_summs(sample)

test = Z[-1]
