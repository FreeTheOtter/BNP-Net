import networkx as nx
import igraph as ig
import numpy as np
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ig.plot(G, layout=layout, target=ax)

g = ig.Graph.Read_GML('celegansneural.gml')

X = np.matrix(g.get_adjacency().data)


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

n = 3
nn = [_ for _ in range(N)]
nn.remove(n)

K = len(z[0])

m = np.atleast_2d(np.sum(z[nn,:], 0)).T
M = np.tile(m, (1, K))

X_ = X[np.ix_(nn,nn)]

M1 = z[nn,:].T * X_ * z[nn,:] - \
    np.diag(np.sum(X_,0) )


M1 = z(nn,:)'*X(nn,nn)*z(nn,:)- ... % No. of links between components
diag(sum(X(nn,nn)*z(nn,:).*z(nn,:))/2);