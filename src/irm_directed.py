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

n = 0
#nn = index mask without currently sampled node n
nn = [_ for _ in range(N)]  
nn.remove(n) 

X_ = X[np.ix_(nn,nn)] #adjacency matrix without currently sampled node

# K = n. of components
K = len(z[0]) 

# m = n. of nodes in each component 
m = np.sum(z[nn,:], 0)[np.newaxis]
M = np.tile(m, (K, 1))


# M1 = n. of links between components without current node
M1 = z[nn,:].T @ X_ @ z[nn,:] - np.diag(np.sum(X_@z[nn,:]*z[nn,:], 0) / 2) 

M1 = z[nn,:].T @ X_ @ z[nn,:]

# M0 = n. of non-links between components without current node
M0 = m.T@m - np.diag((m*(m+1) / 2).flatten()) - M1 

M0 = m*(m-1) - M1

# r = n. of links from current node to components
r = z[nn,:].T @ X[n, nn]
R = np.tile(r, (K, 1))

# s = n. of links from components to current node
s = z[nn,:].T @ X[nn, n]
S = np.tile(s, (1, K))

# # lik matrix of current node sampled to each component
# likelihood = betaln(M1+R+a, M0+M-R+b) - betaln(M1+a, M0+b)
# # lik of current node to new component
# likelihood_n = betaln(r+a, m-r+b) - betaln(a,b)

likelihood = betaln(M1+R+S+a, M0+(M*2)-R-S+b) - betaln(M1+a, M0+b) 

likelihood_n = betaln(r+s+a, (m*2)-r-s+b) - betaln(a,b)

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








