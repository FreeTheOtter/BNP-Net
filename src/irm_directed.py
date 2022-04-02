import networkx as nx
import igraph as ig
import numpy as np
from scipy.special import betaln

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ig.plot(G, layout=layout, target=ax)

g = ig.Graph.Read_GML('celegansneural.gml')
X_full = np.array(g.get_adjacency().data)

def irm_directed(X, T, a, b, A, random_seed = 42):
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

            # m = n. of nodes in each component 
            m = np.atleast_2d(np.sum(z[nn,:], 0)).T
            M = np.tile(m, (1, K))
            

            # M1 = n. of links between components without current node
            M1 = z[nn,:].T @ X_ @ z[nn,:] - np.diag(np.sum(X_@z[nn,:]*z[nn,:], 0) / 2) 
            
            # M0 = n. of non-links between components without current node
            M0 = m@m.T - np.diag((m*(m+1) / 2).flatten()) - M1 

            # r = n. of links from current node to components
            r = z[nn,:].T @ X[nn, n]
            R = np.tile(np.atleast_2d(r).T, (1, K))

            # lik matrix of current node sampled to each component
            likelihood = np.atleast_2d(betaln(M1+R+a, M0+M-R+b) - betaln(M1+a, M0+b))
            # lik of current node to new component
            likelihood_n = np.atleast_2d(betaln(r+a, m.T-r+b) - betaln(a,b))

            logLik = np.atleast_2d(np.sum(np.concatenate([likelihood, likelihood_n]), 1)).T
            logPrior = np.log(np.concatenate([m, np.atleast_2d(A)]))



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

    print(z)
    print(m)
    return Z 

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

        # m = n. of nodes in each component 
        m = np.atleast_2d(np.sum(z[nn,:], 0)).T
        M = np.tile(m, (1, K))
        

        # M1 = n. of links between components without current node
        M1 = z[nn,:].T @ X_ @ z[nn,:] - np.diag(np.sum(X_@z[nn,:]*z[nn,:], 0) / 2) 
        
        # M0 = n. of non-links between components without current node
        M0 = m@m.T - np.diag((m*(m+1) / 2).flatten()) - M1 

        # r = n. of links from current node to components
        r = z[nn,:].T @ X[nn, n]
        R = np.tile(np.atleast_2d(r).T, (1, K))

        # lik matrix of current node sampled to each component
        likelihood = np.atleast_2d(betaln(M1+R+a, M0+M-R+b) - betaln(M1+a, M0+b))
        # lik of current node to new component
        likelihood_n = np.atleast_2d(betaln(r+a, m.T-r+b) - betaln(a,b))

        logLik = np.atleast_2d(np.sum(np.concatenate([likelihood, likelihood_n]), 1)).T
        logPrior = np.log(np.concatenate([m, np.atleast_2d(A)]))



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

print(z)
print(m)


T = 500
a = 1
b = 1
A = 10
Z = irm(X, 493, a, b, A)

for i in range(1, 11):
    print(np.sum(Z[-i], 0))

g = ig.Graph.Read_GML('celegansneural.gml')
X_full = np.array(g.get_adjacency().data)
X = X_full[:5, :5]

X = np.array([[0, 1, 1, 0, 1],
              [1, 0, 0, 0, 1],
              [1, 0, 0, 1, 1],
              [1, 1, 1, 0, 0],
              [0, 0, 1, 1, 0]])

N = len(X)
z = np.ones([N,1])
Z = []

n = 1
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
M0 = m.T@m - np.diag((m*(m+1) / 2).flatten()) - M1 

M0 = m.T@m - np.diag(m.flatten()) - M1

# r = n. of links from current node to components
r = z[nn,:].T @ X[n, nn]
R = np.tile(r, (K, 1))

# s = n. of links from components to current node
s = z[nn,:].T @ X[nn, n]
S = np.tile(s[np.newaxis].T, (1, K))


M2 = M1.T[~np.eye(M1.T.shape[0],dtype=bool)].reshape(M1.T.shape[0], -1)

link_matrix = np.concatenate([M1,M2],axis=1)

current_node_links = np.zeros((link_matrix.shape[0], link_matrix.shape[1]))
current_node_links[0:R.shape[0], 0:R.shape[1]] += R

s_diag = np.diag(s.flatten())
current_node_links[0:s_diag.shape[0], 0:s_diag.shape[1]] += s_diag

S = S.T[~np.eye(S.shape[0],dtype=bool)].reshape(S.shape[0], -1)
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


