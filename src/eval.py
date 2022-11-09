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

Z1 = Z[-1]


# X = np.array([[0, 1, 1],
#               [1, 0, 0],
#               [1, 0, 0]])

# Z1 = np.array([[0, 1],
#                 [1, 0],
#                 [1, 0]])


# X = np.array([[0, 1, 1, 1, 1],
#               [1, 0, 1, 0, 0],
#               [1, 1, 0, 1, 0],
#               [1, 0, 1, 0, 0],
#               [1, 0, 0, 0, 0]])

# Z1 = np.array([[0, 1, 0],
#                 [1, 0, 0],
#                 [1, 0, 0],
#                 [1, 0, 0],
#                 [0, 0, 1]])

# A = np.array([np.where(Z1[i,:] == 1)[0] for i in range(len(Z1))]).flatten()

# Z1.T @ X @ Z1
# np.diag(np.sum(X@Z1*Z1, 0) / 2) 

# M1 = Z1.T @ X @ Z1 - np.diag(np.sum(X@Z1*Z1, 0) / 2) 

# m = np.sum(Z1, 0)[np.newaxis]
# # M = np.tile(m, (K, 1))

# M0 = m.T@m - np.diag((m*(m+1) / 2).flatten()) - M1 

# rhos = np.zeros((len(X), len(X)))

# for i in range(len(X)):
#     for j in range(len(X)):
#         if i == j:
#             continue
#         links = M1[A[i], A[j]]
#         non_links = M0[A[i], A[j]]
#         rhos[i,j] += (links + a) / (links + non_links + a + b)

def compute_rhos(X, Z1):
    A = np.array([np.where(Z1[i,:] == 1)[0] for i in range(len(Z1))]).flatten()

    Z1.T @ X @ Z1
    np.diag(np.sum(X@Z1*Z1, 0) / 2) 

    M1 = Z1.T @ X @ Z1 - np.diag(np.sum(X@Z1*Z1, 0) / 2) 

    m = np.sum(Z1, 0)[np.newaxis]

    M0 = m.T@m - np.diag((m*(m+1) / 2).flatten()) - M1 

    rhos = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue
            links = M1[A[i], A[j]]
            non_links = M0[A[i], A[j]]
            rhos[i,j] += (links + a) / (links + non_links + a + b)

    return rhos

rhos = np.zeros((len(X), len(X)))
for i in sample:
    rhos += compute_rhos(X, i)
rhos /= len(sample)