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

import igraph as ig
import numpy as np
from scipy.special import betaln

def irm_directed(X, T, a, b, A, random_seed = 42):
    N = len(X)
    z = np.ones([N,1])
    Z = []

    np.random.seed(random_seed)

    for t in range(T): # for T iterations
        print(t)
        for n in range(N): # for each node n
            #nn = index mask without currently sampled node n
            nn = [_ for _ in range(N)]  
            nn.remove(n) 

            
            # Delete empty component if present
            idx = np.argwhere(np.all(z[nn, :] == 0, axis=0))
            z = np.delete(z, idx, axis=1)

            X_ = X[np.ix_(nn,nn)] #adjacency matrix without currently sampled node

            # K = n. of components
            K = len(z[0]) 

            # m = n. of nodes in each component 
            m = np.sum(z[nn,:], 0)[np.newaxis] #newaxis allows m to become 2d array (for transposing)

            # M = max links from other clusts
            M = np.tile(m, (K, 1)) + np.diag(m.flatten())

            # M1 = n. of links between components without current node
            M1 = z[nn,:].T @ X_ @ z[nn,:]

            # M0 = n. of non-links between components without current node
            # M0 = m.T@m - np.diag((m*(m+1) / 2).flatten()) - M1 

            M0 = m.T@m - np.diag(m.flatten()) - M1

            # r = n. of links from current node to components
            r = z[nn,:].T @ X[n, nn]
            R = np.tile(r, (K, 1))

            # s = n. of links from components to current node
            s = z[nn,:].T @ X[nn, n]
            S = np.tile(s[np.newaxis].T, (1, K))


            M2 = M1.T[~np.eye(M1.T.shape[0],dtype=bool)].reshape(M1.T.shape[0], -1).copy()

            link_matrix = np.concatenate([M1,M2],axis=1)

            current_node_links = np.zeros((link_matrix.shape[0], link_matrix.shape[1]))
            current_node_links[0:R.shape[0], 0:R.shape[1]] += R

            s_diag = np.diag(s.flatten())
            current_node_links[0:s_diag.shape[0], 0:s_diag.shape[1]] += s_diag

            S = S.T[~np.eye(S.shape[0],dtype=bool)].reshape(S.shape[0], -1)
            if K > 1: 
                current_node_links[:,-S.shape[1]:] += S

            #
            M0_2 = M0.T[~np.eye(M0.T.shape[0],dtype=bool)].reshape(M0.T.shape[0], -1)
            non_link_matrix = np.concatenate([M0, M0_2], axis=1)

            M__2 = M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0], -1)
            max_links_current_node = np.concatenate([M,M__2],axis=1)


            likelihood = np.sum(betaln(link_matrix+current_node_links+a, non_link_matrix+(max_links_current_node)-current_node_links+b) \
                        - betaln(link_matrix+a, non_link_matrix+b), 1)
            #likelihood = 

            likelihood_n = np.sum(betaln(np.hstack([r,s])+a, np.hstack([m-r,m-s])+b) - betaln(a,b),1)

            logLik = np.concatenate([likelihood, likelihood_n])
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

    for n in range(N): # for each node n
        #nn = index mask without currently sampled node n
        nn = [_ for _ in range(N)]  
        nn.remove(n) 

        X_ = X[np.ix_(nn,nn)] #adjacency matrix without currently sampled node

        # K = n. of components
        K = len(z[0]) 

        # m = n. of nodes in each component 
        m = np.sum(z[nn,:], 0)[np.newaxis] #newaxis allows m to become 2d array (for transposing)

        # M = max links from other clusts
        M = np.tile(m, (K, 1)) + np.diag(m.flatten())

        # M1 = n. of links between components without current node
        M1 = z[nn,:].T @ X_ @ z[nn,:]

        # M0 = n. of non-links between components without current node
        # M0 = m.T@m - np.diag((m*(m+1) / 2).flatten()) - M1 

        M0 = m.T@m - np.diag(m.flatten()) - M1

        # r = n. of links from current node to components
        r = z[nn,:].T @ X[n, nn]
        R = np.tile(r, (K, 1))

        # s = n. of links from components to current node
        s = z[nn,:].T @ X[nn, n]
        S = np.tile(s[np.newaxis].T, (1, K))


        M2 = M1.T[~np.eye(M1.T.shape[0],dtype=bool)].reshape(M1.T.shape[0], -1).copy()

        link_matrix = np.concatenate([M1,M2],axis=1)

        current_node_links = np.zeros((link_matrix.shape[0], link_matrix.shape[1]))
        current_node_links[0:R.shape[0], 0:R.shape[1]] += R

        s_diag = np.diag(s.flatten())
        current_node_links[0:s_diag.shape[0], 0:s_diag.shape[1]] += s_diag

        S = S.T[~np.eye(S.shape[0],dtype=bool)].reshape(S.shape[0], -1)
        if K > 1: 
            current_node_links[:,-S.shape[1]:] += S

        #
        M0_2 = M0.T[~np.eye(M0.T.shape[0],dtype=bool)].reshape(M0.T.shape[0], -1)
        non_link_matrix = np.concatenate([M0, M0_2], axis=1)

        M__2 = M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0], -1)
        max_links_current_node = np.concatenate([M,M__2],axis=1)


        likelihood = np.sum(betaln(link_matrix+current_node_links+a, non_link_matrix+(max_links_current_node)-current_node_links+b) \
                    - betaln(link_matrix+a, non_link_matrix+b), 1)
        #likelihood = 

        likelihood_n = np.sum(betaln(np.hstack([r,s])+a, np.hstack([m-r,m-s])+b) - betaln(a,b),1)

        logLik = np.concatenate([likelihood, likelihood_n])
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


g = ig.Graph.Read_GML('celegansneural.gml')
X = np.array(g.get_adjacency().data)
X[X>1] = 1

X = X[:10, :10]

N = len(X)
z = np.ones([N,1])
Z = []

np.random.seed(42)
T = 249
a = 1
b = 1
A = 20

Z = irm_directed(X, T, a, b, A)



g = ig.Graph.Read_Pajek('datasets/Hi-tech.net')

g = ig.Graph.Read_Pajek('datasets/centrality_literature.net')








