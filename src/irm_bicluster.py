import numpy as np
import igraph as ig
from scipy.special import betaln

def irm_bicluster(X, T, a, b, A, set_seed = True, random_seed = 42, print_iter = False):
    N = len(X)
    zr = np.ones([N, 1])
    zc = np.ones([N, 1])
    Zr = []
    Zc = []

    if set_seed:
        np.random.seed(random_seed)

    for t in range(T):
        for n in range(N):
            nn = [_ for _ in range(N)]
            nn.remove(n)

            X_ = X.copy()
            X_[n,:] = 0

            Kr = len(zr[0])

            if Kr > 1:
                idx = np.argwhere(np.sum(zr[nn], 0) == 0)
                zr = np.delete(zr, idx, axis=1)
                Kr -= len(idx)

            # m = n. of nodes in each component 
            mr = np.sum(zr[nn,:], 0)[np.newaxis] #newaxis allows m to become 2d array (for transposing)
            mc = np.sum(zc[nn,:], 0)[np.newaxis]
            Mc = np.tile(mc, (Kr, 1))
            M1 = zr.T @ X_ @ zc

            X_rev = (np.where((X_==0)|(X_==1), X_^1, X_) - np.eye(X_.shape[0])).copy() #reverse matrix for non_links
            X_rev[n,:] = 0
            M0 = zr.T @ X_rev @ zc #n. of non-links between biclusters without current node

            r = zc[nn,:].T @ X[n, nn]
            R = np.tile(r, (Kr, 1))

            logLik_exComp = np.sum(betaln(M1+R+a, M0+Mc-R+b) - betaln(M1+a, M0+b),1)
            logLik_newComp = np.sum(betaln(r+a, mc-r+b) - betaln(a,b),1)

            logLik = np.concatenate([logLik_exComp, logLik_newComp])
            logPrior = np.log(np.append(mr, A))

            logPost = logPrior + logLik

            P = np.exp(logPost-max(logPost)) 

            # Assignment through random draw fron unif(0,1), taking first value from prob. vector
            draw = np.random.rand()
            i = np.argwhere(draw<np.cumsum(P)/sum(P))[0]

            zr[n,:] = 0
            if i == Kr: # If new component: add new column to partition matrix
                zr = np.hstack((zr, np.zeros((N,1)))) 
            zr[n,i] = 1

        for n in range(N):
            nn = [_ for _ in range(N)]
            nn.remove(n)

            X_ = X.copy()
            X_[:,n] = 0

            Kc = len(zc[0])

            if Kc > 1:
                idx = np.argwhere(np.sum(zc[nn], 0) == 0)
                zc = np.delete(zc, idx, axis=1)
                Kc -= len(idx)

            # m = n. of nodes in each component 
            mr = np.sum(zr[nn,:], 0)[np.newaxis] #newaxis allows m to become 2d array (for transposing)
            mc = np.sum(zc[nn,:], 0)[np.newaxis]
            Mr = np.tile(mr.T, (1, Kc))

            M1 = zr.T @ X_ @ zc

            X_rev = (np.where((X_==0)|(X_==1), X_^1, X_) - np.eye(X_.shape[0])).copy() #reverse matrix for non_links
            X_rev[:,0] = 0
            M0 = zr.T @ X_rev @ zc #n. of non-links between biclusters without current node

            s = zr[nn,:].T @ X[nn, n]
            S = np.tile(s[np.newaxis].T, (1, Kc))

            logLik_exComp = np.sum(betaln(M1+S+a, M0+Mr-S+b) - betaln(M1+a, M0+b), 0)
            logLik_newComp = np.sum(betaln(s+a, mr-s+b) - betaln(a,b),1)

            logLik = np.concatenate([logLik_exComp, logLik_newComp])
            logPrior = np.log(np.append(mc, A))

            logPost = logPrior + logLik

            P = np.exp(logPost-max(logPost)) 

            # Assignment through random draw fron unif(0,1), taking first value from prob. vector
            draw = np.random.rand()
            i = np.argwhere(draw<np.cumsum(P)/sum(P))[0]

            zc[n,:] = 0
            if i == Kc: # If new component: add new column to partition matrix
                zc = np.hstack((zc, np.zeros((N,1)))) 
            zc[n,i] = 1

        Zr.append(zr.copy())
        Zc.append(zc.copy())
    
    return np.array([Zr, Zc])
