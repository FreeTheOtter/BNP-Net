import igraph as ig
import numpy as np
from scipy.special import betaln

# g = ig.Graph.Read_GML('karate.txt')
# X = np.array(g.get_adjacency().data)

def irm(X, T, a, b, A, random_seed = 42):
    N = len(X)
    z = np.ones([N,1])
    Z = []

    np.random.seed(random_seed)

    for t in range(T): # for T iterations
        for n in range(N): # for each node n

            #nn = index mask without currently sampled node n
            nn = [_ for _ in range(N)]  
            nn.remove(n) 

            X_ = X[np.ix_(nn,nn)] #adjacency matrix without currently sampled node

            # K = n. of components
            K = len(z[0]) 

            # Delete empty component if present
            if K > 1:
                idx = np.argwhere(np.sum(z[nn], 0) == 0)
                z = np.delete(z, idx, axis=1)
                K -= len(idx)

            # m = n. of nodes in each component 
            m = np.sum(z[nn], 0)[np.newaxis]
            M = np.tile(m, (K, 1))
            

            # M1 = n. of links between components without current node
            M1 = z[nn].T @ X_ @ z[nn] - np.diag(np.sum(X_@z[nn]*z[nn], 0) / 2) 
            
            # M0 = n. of non-links between components without current node
            M0 = m.T@m - np.diag((m*(m+1) / 2).flatten()) - M1 

            # r = n. of links from current node to components
            r = z[nn].T @ X[nn, n]
            R = np.tile(r, (K, 1))

            # lik matrix of current node sampled to each component
            likelihood = betaln(M1+R+a, M0+M-R+b) - betaln(M1+a, M0+b)
            # lik of current node to new component
            likelihood_n = betaln(r+a, m-r+b) - betaln(a,b)

            logLik = np.sum(np.concatenate([likelihood, likelihood_n]), 1)
            logPrior = np.log(np.append(m, A))

            logPost = logPrior + logLik

            # Convert from log probabilities, normalized to max
            P = np.exp(logPost-max(logPost)) 

            # Assignment through random draw fron unif(0,1), taking first value from prob. vector
            draw = np.random.rand()
            i = np.argwhere(draw<np.cumsum(P)/sum(P))[0]

            # Assignment of current node to component i
            z[n,:] = 0
            if i == K: # If new component: add new column to partition matrix
                z = np.hstack((z, np.zeros((N,1)))) 
            z[n,i] = 1

        # Delete empty component if present
        idx = np.argwhere(np.all(z[..., :] == 0, axis=0))
        z = np.delete(z, idx, axis=1)    

        Z.append(z)

    return Z 

# T = 500
# a = 1
# b = 1
# A = 10
# Z = irm(X, T, a, b, A)

# for i in range(1, 11):
#     print(np.sum(Z[-i], 0))






