import numpy as np

def generate_graph(S, thetas = 'random', type = 'undirected', ret_thetas = False, random_seed = None):
    N = np.sum(S)
    K = len(S)

    H = []
    c = 0
    for i in S:
        for j in range(i):
            H += [c]
        c+=1

    if isinstance(random_seed, int):
        np.random.seed(random_seed)
        
    if isinstance(thetas, str) and thetas == 'random':
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

def generate_graph_sep(S1, S2, thetas1 = 'random', thetas2 = 'random', ret_thetas = False, random_seed = None):
    K1 = len(S1)
    K2 = len(S2)
    if isinstance(random_seed, int):
        np.random.seed(random_seed)
        
    if isinstance(thetas1, str) and thetas1 == 'random':
        thetas1 = np.zeros((K1,K1))
        for i in range(K1):
            for j in range(K1):
                thetas1[i, j] = np.random.uniform()
        trThetas1 = np.triu(thetas1)
        thetas1 = np.where(trThetas1, trThetas1, trThetas1.T)
        

    if isinstance(thetas2, str) and thetas2 == 'random':
        thetas2 = np.zeros((K2,K2))
        for i in range(K2):
            for j in range(K2):
                thetas2[i, j] = np.random.uniform()
        trThetas2 = np.tril(thetas2)
        thetas2 = np.where(trThetas2, trThetas2, trThetas2.T)

    X_upper = np.triu(generate_graph(S1, thetas = thetas1, type = 'undirected', ret_thetas = False, random_seed = random_seed))
    X_lower = np.tril(generate_graph(S2, thetas = thetas2, type = 'undirected', ret_thetas = False, random_seed = random_seed))

    X = X_upper + X_lower

    if ret_thetas:
        return X, thetas1, thetas2
    else:
        return X



    










