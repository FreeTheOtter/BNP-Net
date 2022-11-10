import igraph as ig
import numpy as np
from scipy.special import betaln
from irm_undirected import irm
from irm_directed import irm_directed

def irm_directed_separate(X, T, a, b, A, random_seed = 42):
    N = len(X)
    z = np.ones([N,1])
    Z = []

    np.random.seed(random_seed)

    X_upper = np.triu(X)
    X_upper = np.where(X_upper, X_upper, X_upper.T) #make it symmetric

    Z_outgoing = irm(X_upper, T, a, b, A, random_seed)

    X_lower = np.tril(X)
    X_lower = np.where(X_lower, X_lower, X_lower.T) #make it symmetric

    Z_incoming = irm(X_lower, T, a, b, A, random_seed)
    return Z_outgoing, Z_incoming


# g = ig.Graph.Read_GML('celegansneural.gml')
# X = np.array(g.get_adjacency().data)
# X[X>1] = 1


# X = X[:10,:10]


# N = len(X)
# z = np.ones([N,1])
# Z = []

# np.random.seed(42)
# T = 500
# a = 1
# b = 1
# A = 20

# Z_out, Z_in = irm_directed_separate(X, T, a, b, A)

# Z_directed = irm_directed(X, T, a, b, A)

# sample = retrieve_samples(Z_directed)

