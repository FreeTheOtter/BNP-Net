import numpy as np
from irm_undirected import irm_undirected

def irm_directed_separate(X, T, a, b, A, set_seed = True, random_seed = 42, print_iter = False):
    N = len(X)
    z = np.ones([N,1])
    Z = []

    if set_seed:
        np.random.seed(random_seed)

    X_upper = np.triu(X)
    X_upper = np.where(X_upper, X_upper, X_upper.T) #make it symmetric

    Z_outgoing = irm_undirected(X_upper, T, a, b, A, set_seed, random_seed, print_iter)

    X_lower = np.tril(X)
    X_lower = np.where(X_lower, X_lower, X_lower.T) #make it symmetric

    Z_incoming = irm_undirected(X_lower, T, a, b, A, set_seed, random_seed, print_iter)
    return [Z_outgoing, Z_incoming]


