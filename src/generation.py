import numpy as np

def generate_graph(S, thetas = 'random', type = 'undirected', ret_thetas = False):
    N = np.sum(S)
    K = len(S)

    H = []
    c = 0
    for i in S:
        for j in range(i):
            H += [c]
        c+=1

    if np.all(thetas) == 'random':
        thetas = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                thetas[i, j] = np.random.uniform()
        
    if type == 'undirected':
        trThetas = np.triu(thetas)
        thetas = np.where(trThetas, trThetas, trThetas.T)

    #TODO: assert theta valid

    X = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            X[i, j] = np.random.binomial(1, thetas[H[i], H[j]]) #Draw edge from theta cluster probabilities

    if type == 'undirected':
        trX = np.triu(X)
        X = np.where(trX, trX, trX.T) #make X symmetric
    
    if ret_thetas:
        return X, thetas
    else:
        return X










