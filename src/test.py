import networkx as nx
import igraph as ig
import numpy as np
from scipy.special import betaln

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ig.plot(G, layout=layout, target=ax)

g = ig.Graph.Read_GML('celegansneural.gml')
g = ig.Graph.Read_GML('karate.txt')

X = np.array(g.get_adjacency().data)


def irm(X, T, a, b, A):
    N = len(X[0])
    z = np.ones([N,1])
    Z = np.zeros([T,1])

    idx = [_ for _ in range(N)]
    for t in range(T):
        for n in range(N):
            nn = [_ for _ in range(N)]
            nn.remove(n)

            K = len(z[0])

            m = np.sum()

X = X[np.ix_([0,1,2,3,4],[0,1,2,3,4])]
N = len(X)
T = 5
z = np.ones([N,1])
Z = [None]*T

n = 0
nn = [_ for _ in range(N)]
nn.remove(n)

K = len(z[0])

m = np.atleast_2d(np.sum(z[nn,:], 0)).T

M = np.tile(m, (1, K))

X_ = X[np.ix_(nn,nn)]

M1 = z[nn,:].T @ X_ @ z[nn,:] - \
    np.diag(np.sum(X_@z[nn,:]*z[nn,:], 0) / 2)

M0 = m@m.T - np.diag((m*(m+1) / 2).flatten()) - M1

r = z[nn,:].T @ X[nn, n]
R = np.tile(np.atleast_2d(r).T, (1, K))

a = 1
b = 1
A = 10

prior = np.atleast_2d(betaln(M1+R+a, M0+M-R+b) - betaln(M1+a, M0+b))
prior_update = np.atleast_2d(betaln(r+a, m.T-r+b) - betaln(a,b))

logP = np.atleast_2d(np.sum(np.concatenate([prior, prior_update]), 1)).T + \
    np.log(np.concatenate([m, np.atleast_2d(A)]))

P = np.exp(logP-max(logP)) 
print(P)
draw = np.random.rand()
i = np.argwhere(draw<np.cumsum(P)/sum(P))[0]
print(i)

z[n,:] = 0
if i == K:
    z = np.hstack((z, np.zeros((N,1)))) 
z[n,i] = 1


print(z)

idx = np.argwhere(np.all(z[..., :] == 0, axis=0))

z2 = np.delete(z, idx, axis=1)
"""
[5., 3., 8., 3., 7., 8.]) #A = 10
[7., 6., 7., 4., 3., 7.]) #A = 5
[3., 7., 6., 7., 4., 7.]) #A = 1
"""











