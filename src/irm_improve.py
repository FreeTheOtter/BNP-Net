import igraph as ig
import numpy as np
from scipy.special import betaln

g = ig.Graph.Read_GML('karate.txt')
X = np.array(g.get_adjacency().data)

def irm(X, T, a, b, A, random_seed = 42):
    N = len(X)
    z = np.ones([N,1])
    Z = []

    np.random.seed(random_seed)

    for t in range(T):
        for n in range(N):
            nn = [_ for _ in range(N)]  
            nn.remove(n)

            K = len(z[0])

            m = np.atleast_2d(np.sum(z[nn,:], 0)).T
            print(m)
            M = np.tile(m, (1, K))

            X_ = X[np.ix_(nn,nn)]

            M1 = z[nn,:].T @ X_ @ z[nn,:] - \
                np.diag(np.sum(X_@z[nn,:]*z[nn,:], 0) / 2)

            M0 = m@m.T - np.diag((m*(m+1) / 2).flatten()) - M1

            r = z[nn,:].T @ X[nn, n]
            R = np.tile(np.atleast_2d(r).T, (1, K))

            prior = np.atleast_2d(betaln(M1+R+a, M0+M-R+b) - betaln(M1+a, M0+b))
            prior_n = np.atleast_2d(betaln(r+a, m.T-r+b) - betaln(a,b))

            logPrior = np.atleast_2d(np.sum(np.concatenate([prior, prior_n]), 1)).T
            logLik = np.log(np.concatenate([m, np.atleast_2d(A)]))
            logPost = logPrior + logLik

            P = np.exp(logPost-max(logPost)) 

            draw = np.random.rand()
            i = np.argwhere(draw<np.cumsum(P)/sum(P))[0]

            z[n,:] = 0
            if i == K:
                z = np.hstack((z, np.zeros((N,1)))) 
            z[n,i] = 1

            idx = np.argwhere(np.all(z[..., :] == 0, axis=0))
            z = np.delete(z, idx, axis=1)

        Z.append(z)
    return Z




T = 500
a = 1
b = 1
A = 10
Z = irm(X, T, a, b, A)

for i in range(1, 11):
    print(np.sum(Z[-i], 0))






