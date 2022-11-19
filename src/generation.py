import numpy as np
import random

def generate_graph(N, S, thetas = 'random', type = 'undirected'):
    K = len(S)
    H = []
    c = 0
    for i in S:
        for j in range(i):
            H += [c]
        c+=1

    if type == 'undirected':
        if thetas == 'random':
            thetas = np.zeros((K,K))
            for i in range(K):
                for j in range(K):
                    thetas[i, j] = np.random.uniform()
            trThetas = np.triu(thetas)
            thetas = np.where(trThetas, trThetas, trThetas.T)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                X[i, j] = np.random.binomial(1, thetas[H[i], H[j]]) #Draw edge from theta cluster probabilities

        trX = np.triu(X)
        X = np.where(trX, trX, trX.T) #make X symmetric
    return X

N = 10
X = np.zeros((N,N))
K = 2
S = [4, 6] #Z collapsed, ordered

H = []
c = 0
for i in S:
    for _ in range(i):
        H += [c]
    c+=1
    
thetas = np.array([[0.5, 0.8],
                    [0.8, 0.3]])


for i in range(N):
    for j in range(N):
        if i == j:
            continue
        X[i, j] = np.random.binomial(1, thetas[H[i], H[j]])

trX = np.triu(X)
X = np.where(trX, trX, trX.T)











