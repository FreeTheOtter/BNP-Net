import numpy as np
from scipy.special import betaln
from scipy.special import gammaln
from igraph.clustering import compare_communities
# from arviz import waic
from scipy.stats import binom

class SBM():
    """
    Main Class for BNP SBM modeling
    """
    
    def __init__(self, setup, start_z = None,
                prior_r = "DP", prior_c = "DP",
                alpha_PY_r = None, alpha_PY_c = None,
                sigma_PY_r = None, sigma_PY_c = None,
                beta_DM_r = None, beta_DM_c = None, K_DM_r = None, K_DM_c = None,
                gamma_GN_r = None, gamma_GN_c = None,
                a = 1, b = 1, set_seed = False
                ):

        self.directed = setup['directed']
        self.binary = setup['binary']
        self.unicluster = setup['unicluster']

        self.start_z = start_z

        self.prior_r = prior_r
        self.prior_c = prior_c

        self.alpha_PY_r = alpha_PY_r
        self.alpha_PY_c = alpha_PY_c
        self.sigma_PY_r = sigma_PY_r
        self.sigma_PY_c = sigma_PY_c
        self.beta_DM_r = beta_DM_r
        self.beta_DM_c = beta_DM_c
        self.K_star_DM_r = K_DM_r
        self.K_star_DM_c = K_DM_c
        self.gamma_GN_r = gamma_GN_r
        self.gamma_GN_c = gamma_GN_c
        self.a = a
        self.b = b

        self.checkPriorParameters()

        if isinstance(set_seed, int):
            np.random.seed(set_seed)

    def evalPrior(self, nn, direction = "rows"):
        if self.unicluster:
            if self.prior_r == "DP":
                return self.prior_DP(np.sum(self.z[nn,:], 0)[np.newaxis], self.alpha_PY_r)
            elif self.prior_r == "PY":
                return self.prior_PY(np.sum(self.z[nn,:], 0)[np.newaxis], self.alpha_PY_r, self.sigma_PY_r)
            elif self.prior_r == "DM":
                return self.prior_DM(np.sum(self.z[nn,:], 0)[np.newaxis], self.beta_DM_r, self.K_star_DM_r)
            elif self.prior_r == "GN":
                return self.prior_GN(np.sum(self.z[nn,:], 0)[np.newaxis], self.gamma_GN_r)
        else:
            if direction == "rows":
                if self.prior_r == "DP":
                    return self.prior_DP(np.sum(self.zr[nn,:], 0)[np.newaxis], self.alpha_PY_r)
                elif self.prior_r == "PY":
                    return self.prior_PY(np.sum(self.zr[nn,:], 0)[np.newaxis], self.alpha_PY_r, self.sigma_PY_r)
                elif self.prior_r == "DM":
                    return self.prior_DM(np.sum(self.zr[nn,:], 0)[np.newaxis], self.beta_DM_r, self.K_star_DM_r)
                elif self.prior_r == "GN":
                    return self.prior_GN(np.sum(self.zr[nn,:], 0)[np.newaxis], self.gamma_GN_r)
            else:
                if self.prior_c == "DP":
                    return self.prior_DP(np.sum(self.zc[nn,:], 0)[np.newaxis], self.alpha_PY_c)
                elif self.prior_c == "PY":
                    return self.prior_PY(np.sum(self.zc[nn,:], 0)[np.newaxis], self.alpha_PY_c, self.sigma_PY_c)
                elif self.prior_c == "DM":
                    return self.prior_DM(np.sum(self.zc[nn,:], 0)[np.newaxis], self.beta_DM_c, self.K_star_DM_c)
                elif self.prior_c == "GN":
                    return self.prior_GN(np.sum(self.zc[nn,:], 0)[np.newaxis], self.gamma_GN_c)

    def fit(self, X, T):
        self.X = X

        X_temp = X.copy().astype(int)
        self.X_rev = (np.where((X_temp==0)|(X_temp==1), X_temp^1, X_temp) - np.eye(X_temp.shape[0])).copy()
        del X_temp

        self.X_bin = self.X.copy()
        self.X_bin[self.X_bin>0] = 1

        self.T = T
        self.N = len(self.X)

        self.X_full = np.ones((self.N, self.N))
        np.fill_diagonal(self.X_full, 0)

        if self.unicluster:
            self.gibbs_unicluster()
        else:
            self.gibbs_bicluster()

    def gibbs_unicluster(self):
        #Different initializations
        if self.start_z == "singleton":
            self.z = np.eye(self.N)
        elif self.start_z is not None:
            # assert correct z
            self.z = self.start_z
        else:
            self.z = np.ones([self.N, 1]) #Init as single cluster

        self.Z = [] #Empty list init

        self.idx_list = [x for x in range(self.N)]
        for _ in range(self.T):
            for n in range(self.N):
                nn = list(self.idx_list)
                nn.remove(n) #nn = index mask without currently sampled node n

                # X_ = self.X[np.ix_(nn,nn)] #adjacency matrix without currently sampled node

                K = len(self.z[0])  # K = n. of components

                # Delete empty component if present
                if K > 1:
                    idx = np.argwhere(np.sum(self.z[nn], 0) == 0)
                    self.z = np.delete(self.z, idx, axis=1)
                    K -= len(idx)

                logLikelihood = self.evalLikelihood(nn, n, K)

                logPrior = np.log(self.evalPrior(nn, self.unicluster))
                
                logPosterior = logPrior + logLikelihood

                # Convert from log probabilities, normalized to max
                p = np.exp(logPosterior-max(logPosterior)) 
                # Assignment through random draw fron unif(0,1), taking first value from prob. vector
                draw = np.random.rand()
                i = np.argwhere(draw<np.cumsum(p)/sum(p))[0]
                

                # Assignment of current node to component i
                self.z[n,:] = 0
                if i == K: # If new component: add new column to partition matrix
                    self.z = np.hstack([self.z, np.zeros((self.N,1))]) 
                self.z[n,i] = 1

                # self.gibbs_sweep(n, directed, binary)

            self.Z.append(self.z.copy())

    def gibbs_bicluster(self):
        #Different initializations
        if self.start_z == "singleton":
            self.zr = np.eye(self.N)
            self.zc = np.eye(self.N)
        elif self.start_z is not None:
            # assert correct z
            self.zr, self.zc = self.start_z[0], self.start_z[1]
        else:
            self.zr = np.ones([self.N, 1])
            self.zc = np.ones([self.N, 1]) #Init as single cluster
        
        self.Z = []
        self.Zr = []
        self.Zc = []

        self.idx_list = [x for x in range(self.N)]

         
        
        for _ in range(self.T):
            for n in range(self.N):
                nn = list(self.idx_list)
                nn.remove(n)

                Kr = len(self.zr[0])

                if Kr > 1:
                    idx = np.argwhere(np.sum(self.zr[nn], 0) == 0)
                    self.zr = np.delete(self.zr, idx, axis=1)
                    Kr -= len(idx)

                logLikelihood = self.evalLikelihoodBicluster(nn, n, Kr, "rows")

                logPrior = np.log(self.evalPrior(nn, direction = "rows"))

                logPosterior = logPrior + logLikelihood

                p = np.exp(logPosterior-max(logPosterior)) 

                # Assignment through random draw fron unif(0,1), taking first value from prob. vector
                draw = np.random.rand()
                i = np.argwhere(draw<np.cumsum(p)/sum(p))[0]

                self.zr[n,:] = 0
                if i == Kr: # If new component: add new column to partition matrix
                    self.zr = np.hstack([self.zr, np.zeros((self.N, 1))]) 
                self.zr[n,i] = 1

            for n in range(self.N):
                nn = list(self.idx_list)
                nn.remove(n)

                Kc = len(self.zc[0])

                if Kc > 1:
                    idx = np.argwhere(np.sum(self.zc[nn], 0) == 0)
                    self.zc = np.delete(self.zc, idx, axis=1)
                    Kc -= len(idx)     

                logLikelihood = self.evalLikelihoodBicluster(nn, n, Kc, "columns")   

                logPrior = np.log(self.evalPrior(nn, direction = "columns"))

                logPosterior = logPrior + logLikelihood

                p = np.exp(logPosterior-max(logPosterior)) 

                # Assignment through random draw fron unif(0,1), taking first value from prob. vector
                draw = np.random.rand()
                i = np.argwhere(draw<np.cumsum(p)/sum(p))[0]

                self.zc[n,:] = 0
                if i == Kc: # If new component: add new column to partition matrix
                    self.zc = np.hstack([self.zc, np.zeros((self.N,1))]) 
                self.zc[n,i] = 1      
                

                         

            self.Z.append([self.zr.copy(), self.zc.copy()])
            self.Zr.append(self.zr.copy())
            self.Zc.append(self.zc.copy())

    # def gibbs_bicluster(self):
    #     self.zr = np.ones([self.N, 1])
    #     self.zc = np.ones([self.N, 1])
    #     self.Z = []
    #     self.Zr = []
    #     self.Zc = []

    #     self.idx_list = [x for x in range(self.N)]
        
    #     for _ in range(self.T):
    #         for n in range(self.N):
    #             self.gibbs_sweep_bicluster(n, "rows", self.directed, self.binary)
    #         for n in range(self.N):
    #             self.gibbs_sweep_bicluster(n, "columns", self.directed, self.binary)

    #         self.Z.append([self.zr.copy(), self.zc.copy()])
    #         self.Zr.append(self.zr.copy())
    #         self.Zc.append(self.zc.copy())
    
    def evalLikelihood(self, nn, n, K):
        X_ = self.X[np.ix_(nn,nn)] #adjacency matrix without currently sampled node

        if self.directed:
            if self.binary: #directed binary
                # m = n. of nodes in each component 
                m = np.sum(self.z[nn,:], 0)[np.newaxis] #newaxis allows m to become 2d array (for transposing)
                
                M1 = self.z[nn,:].T @ X_ @ self.z[nn,:] #n. of links between components without current node

                # r = n. of links from current node to components
                r = self.z[nn,:].T @ self.X[n, nn]
                R = np.tile(r, (K, 1))

                # s = n. of links from components to current node
                s = self.z[nn,:].T @ self.X[nn, n]
                S = np.tile(s[np.newaxis].T, (1, K))

                M2 = M1.T[~np.eye(M1.T.shape[0],dtype=bool)].reshape(M1.T.shape[0], -1).copy()
                LM = np.concatenate([M1,M2],axis=1) #Link Matrix

                LM_n = np.zeros((LM.shape[0], LM.shape[1])) #LM of current node n
                LM_n[0:R.shape[0], 0:R.shape[1]] += R
                s_diag = np.diag(s.flatten())
                LM_n[0:s_diag.shape[0], 0:s_diag.shape[1]] += s_diag
                S = S.T[~np.eye(S.shape[0],dtype=bool)].reshape(S.shape[0], -1)
                if K > 1: 
                    LM_n[:,-S.shape[1]:] += S

                M0 = m.T@m - np.diag(m.flatten()) - M1 #n. of non-links between components without current node
                M0_2 = M0.T[~np.eye(M0.T.shape[0],dtype=bool)].reshape(M0.T.shape[0], -1)
                NLM = np.concatenate([M0, M0_2], axis=1) #Non-Link Matrix

                PLM = np.tile(m, (K, 1)) + np.diag(m.flatten()) #Potential links matrix from other clusts
                PLM_2 = PLM[~np.eye(PLM.shape[0],dtype=bool)].reshape(PLM.shape[0], -1)
                P_n = np.concatenate([PLM,PLM_2],axis=1) #Potential links for current node n

                logLikelihood_old = np.sum(betaln(LM + LM_n + self.a, NLM + P_n - LM_n + self.b) \
                                     - betaln(LM + self.a, NLM + self.b)
                                       , 1) #log prob of assigning to existing clusters

                logLikelihood_new = np.sum(betaln(np.hstack([r,s]) + self.a, np.hstack([m-r,m-s]) + self.b) \
                                         - betaln(self.a, self.b)
                                           , 1) #log prob of assigning to new clusters

                logLikelihood = np.concatenate([logLikelihood_old, logLikelihood_new])
                
            else: #directed weighted
                m = np.sum(self.z[nn,:], 0)[np.newaxis]

                r = self.z[nn,:].T @ self.X[n, nn]
                R = np.tile(r, (K, 1))

                s = self.z[nn,:].T @ self.X[nn, n]
                S = np.tile(s[np.newaxis].T, (1, K))

                M1 = self.z[nn,:].T @ X_ @ self.z[nn,:]
                M2 = M1.T[~np.eye(M1.T.shape[0],dtype=bool)].reshape(M1.T.shape[0], -1).copy()

                LM = np.concatenate([M1,M2],axis=1) #Link Matrix
                LM_n = np.zeros((LM.shape[0], LM.shape[1])) #Link Matrix of current node n
                LM_n[0:R.shape[0], 0:R.shape[1]] += R
                s_diag = np.diag(s.flatten())
                LM_n[0:s_diag.shape[0], 0:s_diag.shape[1]] += s_diag
                S = S.T[~np.eye(S.shape[0],dtype=bool)].reshape(S.shape[0], -1)
                if K > 1: 
                    LM_n[:,-S.shape[1]:] += S

                Max = np.tile(m, (K, 1)) + np.diag(m.flatten())
                Max_2 = Max[~np.eye(Max.shape[0],dtype=bool)].reshape(Max.shape[0], -1)
                MLM_n = np.concatenate([Max,Max_2],axis=1) #Max Link Matrix

                C1 = m.T@m - np.diag(m.flatten())
                C2 = C1.T[~np.eye(C1.T.shape[0],dtype=bool)].reshape(C1.T.shape[0], -1).copy()
                C = np.concatenate([C1, C2], axis=1) #no. of possible links

                F = np.zeros((LM.shape[0], LM.shape[1]))
                f_out = np.sum(gammaln(np.multiply(self.z[nn,:].T, self.X[n, nn]) + 1), axis = 1)
                F_out = np.tile(f_out, (K, 1))

                f_in = np.sum(gammaln(np.multiply(self.z[nn,:].T, self.X[nn, n]) + 1), axis = 1)
                f_in_diag = np.diag(f_in)
                F_in = np.tile(f_in[np.newaxis].T, (1, K))
                F_in = F_in.T[~np.eye(F_in.shape[0],dtype=bool)].reshape(F_in.shape[0], -1)

                F[0:F_out.shape[0], 0:F_out.shape[1]] += F_out
                F[0:F_out.shape[0], 0:F_out.shape[1]] += f_in_diag
                if K > 1: 
                    F[:,-F_in.shape[1]:] += F_in
                if K == 1:
                    F[:] += f_in


                logLikelihood_old = np.sum(gammaln(LM + LM_n + self.a) - gammaln(LM + self.a) \
                            + (LM + self.a)*np.log(C + self.b) - (LM + LM_n + self.a)*np.log(C + MLM_n + self.b) \
                            - F, 1)

                logLikelihood_new = np.sum(gammaln(np.hstack([r,s]) + self.a) - gammaln(self.a) \
                            + (self.b)*np.log(self.a) - (np.hstack([r,s]) + self.a)*np.log(np.hstack([m,m]) + self.b) \
                            - np.hstack([f_out, f_in]))[np.newaxis]

                logLikelihood = np.concatenate([logLikelihood_old, logLikelihood_new])            
        else:
            if self.binary: #undirected binary
                # m = n. of nodes in each component 
                m = np.sum(self.z[nn], 0)[np.newaxis]
                P = np.tile(m, (K, 1)) #Potential links

                LM = self.z[nn].T @ X_ @ self.z[nn] - np.diag(np.sum(X_@self.z[nn]*self.z[nn], 0) / 2) #n. of links between components without current node

                NLM = m.T@m - np.diag((m*(m+1) / 2).flatten()) - LM #n. of non-links between components without current node

                r = self.z[nn].T @ self.X[nn, n] #n. of links from current node to components
                LM_n = np.tile(r, (K, 1))

                logLikelihood_old = np.sum(betaln(LM + LM_n + self.a, NLM + P - LM_n + self.b) \
                                         - betaln(LM + self.a, NLM + self.b)
                                           , 1)

                logLikelihood_new = np.sum(betaln(r + self.a, m - r + self.b) \
                                         - betaln(self.a, self.b)
                                           , 1)

                logLikelihood = np.concatenate([logLikelihood_old, logLikelihood_new])

            else: #undirected weighted, not tested
                m = np.sum(self.z[nn,:], 0)[np.newaxis]
                C = np.tile(m, (K, 1)) #Potential links

                LM = self.z[nn,:].T @ X_ @ self.z[nn,:] - np.diag(np.sum(X_@self.z[nn]*self.z[nn], 0) / 2)

                r = self.z[nn,:].T @ self.X[n, nn]
                LM_n = np.tile(r, (K, 1))

                X_bin_ = self.X_bin[np.ix_(nn,nn)]

                r_bin = self.z[nn,:].T @ self.X_bin[n, nn]
                LM_n_bin = np.tile(r_bin, (K, 1))

                f = np.sum(gammaln(np.multiply(self.z[nn,:].T, self.X[n, nn]) + 1), axis = 1)
                F = np.tile(f, (K,1))


                logLikelihood_old = np.sum(gammaln(LM + LM_n + self.a) - gammaln(LM + self.a) \
                            + (LM + self.a)*np.log(C + self.b) - (LM + LM_n + self.a)*np.log(C + LM_n_bin + self.b) \
                            - F, 1)

                logLikelihood_new = np.sum(gammaln(r + self.a) - gammaln(self.a) \
                            + (self.b)*np.log(self.a) - (r + self.a)*np.log(r_bin + self.b) \
                            - f)[np.newaxis]

                logLikelihood = np.concatenate([logLikelihood_old, logLikelihood_new])    

        return logLikelihood
    
    def evalLikelihoodBicluster(self, nn, n, K, direction = "rows"):
        X_ = self.X.copy().astype(int)
        # X_bin_ = self.X_bin.copy().astype(int)
        X_full_ = self.X_full.copy().astype(int)
        if direction == "rows":
            if self.binary: #directed binary
                X_[n,:] = 0 #adj matrix without currently sampled node rows

                mc = np.sum(self.zc[nn,:], 0)[np.newaxis]
                Mc = np.tile(mc, (K, 1))
                LM = self.zr.T @ X_ @ self.zc

                X_rev = (np.where((X_==0)|(X_==1), X_^1, X_) - np.eye(X_.shape[0])).copy() #reverse matrix for non_links
                X_rev[n,:] = 0
                NLM = self.zr.T @ X_rev @ self.zc #n. of non-links between biclusters without current node

                r = self.zc[nn,:].T @ self.X[n, nn]
                R = np.tile(r, (K, 1))

                logLikelihood_old = np.sum(betaln(LM + R + self.a, NLM + Mc - R + self.b) \
                                    - betaln(LM + self.a, NLM + self.b)
                                        , 1)
                logLikelihood_new = np.sum(betaln(r + self.a, mc - r + self.b) \
                                        - betaln(self.a, self.b)
                                        , 1)

                logLikelihood = np.concatenate([logLikelihood_old, logLikelihood_new])

                
            else: #weighted
                X_[n,:] = 0
                # X_bin_[n,:] = 0
                X_full_[n,:] = 0

                LM = self.zr.T @ X_ @ self.zc

                
                r = self.zc[nn,:].T @ self.X[n, nn]
                LM_n = np.tile(r, (K, 1))

                mc = np.sum(self.zc[nn,:], 0)[np.newaxis]
                # r_bin = self.zc[nn,:].T @ self.X_bin[n, nn]
                Mc = np.tile(mc, (K,1))

                # C = self.zr.T @ X_bin_ @ self.zc
                C = self.zr.T @ X_full_ @ self.zc

                f = np.sum(gammaln(np.multiply(self.zc[nn,:].T, self.X[n, nn]) + 1), axis = 1)
                F = np.tile(f, (K, 1))

                logLikelihood_old = np.sum(gammaln(LM + LM_n + self.a) - gammaln(LM + self.a) \
                            + (LM + self.a)*np.log(C + self.b) - (LM + LM_n + self.a)*np.log(C + Mc + self.b) \
                            - F, 1)

                logLikelihood_new = np.sum(gammaln(r + self.a) - gammaln(self.a) \
                            + (self.b)*np.log(self.a) - (r + self.a)*np.log(mc + self.b) \
                            - f)[np.newaxis]

                logLikelihood = np.concatenate([logLikelihood_old, logLikelihood_new])

        elif direction == "columns":
            if self.binary:
                X_[:,n] = 0 #adj matrix without currently sampled node columns

                mr = np.sum(self.zr[nn,:], 0)[np.newaxis] #newaxis allows m to become 2d array (for transposing)
                Mr = np.tile(mr.T, (1, K))

                LM = self.zr.T @ X_ @ self.zc

                X_rev = (np.where((X_==0)|(X_==1), X_^1, X_) - np.eye(X_.shape[0])).copy() #reverse matrix for non_links
                X_rev[:,n] = 0
                NLM = self.zr.T @ X_rev @ self.zc #n. of non-links between biclusters without current node

                s = self.zr[nn,:].T @ self.X[nn, n]
                S = np.tile(s[np.newaxis].T, (1, K))

                logLikelihood_new = np.sum(betaln(LM + S + self.a, NLM + Mr - S + self.b) \
                                        - betaln(LM + self.a, NLM + self.b)
                                        , 0)

                logLikelihood_old = np.sum(betaln(s + self.a, mr - s + self.b) \
                                        - betaln(self.a, self.b)
                                        , 1)

                logLikelihood = np.concatenate([logLikelihood_new, logLikelihood_old])
            
            else:
                X_[:,n] = 0
                # X_bin_[:,n] = 0
                X_full_[:,n] = 0

                LM = self.zr.T @ X_ @ self.zc

                s = self.zr[nn,:].T @ self.X[nn, n]
                LM_n = np.tile(s[np.newaxis].T, (1, K))

                mr = np.sum(self.zr[nn,:], 0)[np.newaxis]
                # s_bin = self.zr[nn,:].T @ self.X_bin[nn, n]
                Mr = np.tile(mr.T, (1, K))

                # C = self.zr.T @ X_bin_ @ self.zc
                C = self.zr.T @ X_full_ @ self.zc

                f = np.sum(gammaln(np.multiply(self.zr[nn,:], self.X[nn, n][np.newaxis].T) + 1), axis = 0)
                F = np.tile(f[np.newaxis].T, (1, K))
                

                logLikelihood_old = np.sum(gammaln(LM + LM_n + self.a) - gammaln(LM + self.a) \
                            + (LM + self.a)*np.log(C + self.b) - (LM + LM_n + self.a)*np.log(C + Mr + self.b) \
                            - F, 0)

                logLikelihood_new = np.sum(gammaln(s + self.a) - gammaln(self.a) \
                            + (self.b)*np.log(self.a) - (s + self.a)*np.log(mr + self.b) \
                            - f)[np.newaxis]

                logLikelihood = np.concatenate([logLikelihood_old, logLikelihood_new])

        return logLikelihood

    def retrieve_samples(self, Z, gap = 25, burn_in = True):
        if burn_in == True:
            burn_in = len(Z)//2   
        return Z[burn_in::gap]

    def compute_VIs(self, C):
        N = len(C)
        VIs = np.zeros((N,N))

        for i in range(N):
            for j in range(i):
                VIs[i,j] = compare_communities(C[i], C[j])

        VIs += VIs.T

        return VIs

    def compute_zhat(self, gap = 1):
        if self.unicluster:
            self.Z_sample = self.retrieve_samples(self.Z, gap)
            C = [np.where(x == 1)[1].tolist() for x in self.Z_sample]
            self.VI = self.compute_VIs(C)
            self.VI = self.VI.sum(0)/self.VI.shape[0]

            idx_min = np.argmin(self.VI) 
            self.z_hat = self.Z_sample[idx_min]
            self.c_hat = np.where(self.z_hat==1)[1]
            #temp
            self.zr_hat = self.z_hat
            self.zc_hat = self.z_hat
            self.cr_hat = self.c_hat
            self.cc_hat = self.c_hat

        else:
            self.Zr_sample = self.retrieve_samples(self.Zr, gap)
            self.Zc_sample = self.retrieve_samples(self.Zc, gap)
            Cr = [np.where(x == 1)[1].tolist() for x in self.Zr_sample]
            Cc = [np.where(x == 1)[1].tolist() for x in self.Zc_sample]

            VIr = self.compute_VIs(Cr)
            VIc = self.compute_VIs(Cc)

            self.VI = (VIr + VIc)/2
            self.VI = self.VI.sum(0)/self.VI.shape[0]

            idx_min = np.argmin(self.VI) 
            self.z_hat = [self.Zr_sample[idx_min], self.Zc_sample[idx_min]]
            self.zr_hat = self.Zr_sample[idx_min]
            self.zc_hat = self.Zc_sample[idx_min]

            self.c_hat = [np.where(self.z_hat[0]==1)[1], np.where(self.z_hat[1]==1)[1]]
            self.cr_hat = np.where(self.z_hat[0]==1)[1]
            self.cc_hat = np.where(self.z_hat[1]==1)[1]

    def evalZ(self, z0, gap = 1, alpha_ball = 0.95):
        self.compute_zhat(gap)
        if self.unicluster:
            self.Zr_sample = self.Z_sample
            self.Zc_sample = self.Z_sample
        
        Cr = [np.where(x == 1)[1].tolist() for x in self.Zr_sample]
        Cc = [np.where(x == 1)[1].tolist() for x in self.Zc_sample]

        VIr_true = 0
        VIc_true = 0
        for i in range(len(Cr)):
            VIr_true += compare_communities(Cr[i], z0[0])
            VIc_true += compare_communities(Cc[i], z0[1])

        VIr_true = VIr_true/len(Cr)
        VIc_true = VIc_true/len(Cc)

        self.VI_Z_z0 = np.mean([VIr_true, VIc_true])
        # VI_true = (VIr_true + VIc_true)/2
        # VI_true = VI_true.sum(0)/VI_true.shape[0]

        sorted_Zr = [self.Zr_sample[i] for i in np.argsort(self.VI)]
        sorted_Zc = [self.Zc_sample[i] for i in np.argsort(self.VI)]

        b_threshold = int(len(self.VI)*alpha_ball)

        zr_ball = sorted_Zr[b_threshold]
        zc_ball = sorted_Zc[b_threshold]
        c_ball = [np.where(zr_ball==1)[1], np.where(zc_ball==1)[1]]

        self.VI_Z_zb = np.mean([compare_communities(self.cr_hat, c_ball[0]), 
                                compare_communities(self.cc_hat, c_ball[1])])

        self.VIr_true = compare_communities(self.cr_hat, z0[0])
        self.VIc_true = compare_communities(self.cc_hat, z0[1])

        print("VI_rel_true: ", self.VI_Z_z0)
        print("VI_rel_ball: ", self.VI_Z_zb)

        Kr_vector = [i.shape[1] for i in sorted_Zr]
        Kc_vector = [i.shape[1] for i in sorted_Zc] 

        print("Kr")
        print('0.25: ', np.quantile(np.sort(Kr_vector), 0.25))
        print('0.50: ', np.quantile(np.sort(Kr_vector), 0.5))
        print('0.75: ', np.quantile(np.sort(Kr_vector), 0.75))

        print("Kc")
        print('0.25: ', np.quantile(np.sort(Kc_vector), 0.25))
        print('0.50: ', np.quantile(np.sort(Kc_vector), 0.5))
        print('0.75: ', np.quantile(np.sort(Kc_vector), 0.75))

    def compute_block_probabilities(self, gap=1):
        self.compute_zhat(gap)
        if self.unicluster:
            if self.binary:
                self.block_links = self.z_hat.T @ self.X @ self.z_hat
                self.block_nonlinks = self.z_hat.T @ self.X_rev @ self.z_hat
                # m = np.array(np.sum(self.z_hat, axis = 0))[np.newaxis]
                # self.block_possible_edges = np.outer(m, m)
        else:
            if self.binary:
                self.block_links = self.zr_hat.T @ self.X @ self.zc_hat
                self.block_nonlinks = self.zr_hat.T @ self.X_rev @ self.zc_hat
                # mr = np.array(np.sum(self.zr_hat, axis = 0))
                # mc = np.array(np.sum(self.zc_hat, axis = 0))
                # self.block_possible_edges = np.outer(mr, mc)
        self.estimated_theta = (self.block_links + self.a)/(self.block_links + self.block_nonlinks + self.b)

    def predict(self, gap = 1):
        self.compute_block_probabilities(gap)
        self.X_pred_theta = np.zeros(self.X.shape)

        if self.unicluster:
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[1]):
                    if i == j:
                        continue
                    self.X_pred_theta[i,j] = self.estimated_theta[int(self.c_hat[i]), int(self.c_hat[j])]
        else:
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[1]):
                    if i == j:
                        continue
                    self.X_pred_theta[i,j] = self.estimated_theta[int(self.cr_hat[i]), int(self.cc_hat[j])]
    
    def evalLogLikelihood_full(self, zr = "none", zc = "none", ret_sum = True, ret_waic = False):

        if self.unicluster:
            if zr == "none":
                z = self.z
            else:
                z = zr

            if self.directed:
                if self.binary: #directed binary
                    LM = z.T @ self.X @ z
                    NLM = z.T @ self.X_rev @ z
                else: #directed weighted
                    pass
            else:
                if self.binary: #undirected binary
                    pass
                else: #undirected weighted
                    pass
        else:
            if zr == "none":
                zr = self.zr
            if zc == "none":
                zc = self.zc

            if self.binary: #biclustering binary
                LM = zr.T @ self.X @ zc
                NLM = zr.T @ self.X_rev @ zc
            else: #biclustering weighted
                pass
        if ret_sum:
            logLikelihood = np.sum(betaln(LM + self.a, NLM + self.b) - betaln(self.a, self.b))
        else:
            logLikelihood = betaln(LM + self.a, NLM + self.b) - betaln(self.a, self.b)
        return logLikelihood
    
    # def compute_waic(self, Zsample):
    #     ll = binom.logpmf(self.X, 1, self.X_pred_theta)
    #     lppd = np.sum(ll)
    #     var = np.var(ll)
    #     return -2*lppd + 2*var

    def prior_DP(self, m, alpha):
        return np.append(m, alpha)
    
    def prior_PY(self, m, alpha, sigma):
        return np.append(m-sigma, alpha+len(m)*sigma)

    def prior_DM(self, m, beta, K_hat):
        return np.append(m + beta, beta * (K_hat - len(m)) * (K_hat > len(m)))
    
    def prior_GN(self, m , gamma):
        return np.append((m + 1) * (np.sum(m) - len(m) + gamma), len(m)**2 - len(m)*gamma)
    
    def checkPriorParameters(self):
        # Sanity check if the correct parameters have been inputted for the desired prior
        if self.unicluster:
            if self.prior_r == "DP":
                assert (self.alpha_PY_r is not None), "alpha_PY_r missing for DP prior"
            if self.prior_r == "PY":
                assert (self.alpha_PY_r is not None), "alpha_PY_r missing for PY prior"
                assert (self.sigma_PY_r is not None), "sigma_PY_r missing for PY prior"
                #Potential check for sigma outside of range
            if self.prior_r == "DM":
                assert (self.beta_DM_r is not None), "beta_DM_r missing for DM prior"
                assert (self.K_star_DM_r is not None), "K_star_r missing for DM prior"
            if self.prior_r == "GN":
                assert (self.gamma_GN_r is not None), "gamma_GN_r missing for GN prior"
        else:
            if self.prior_r == "DP":
                assert (self.alpha_PY_r is not None), "alpha_PY_r missing for DP row prior"
            if self.prior_r == "PY":
                assert (self.alpha_PY_r is not None), "alpha_PY_r missing for PY row prior"
                assert (self.sigma_PY_r is not None), "sigma_PY_r missing for PY row prior"
                #Potential check for sigma outside of range
            if self.prior_r == "DM":
                assert (self.beta_DM_r is not None), "beta_DM_r missing for DM row prior"
                assert (self.K_star_DM_r is not None), "K_star_r missing for DM row prior"
            if self.prior_r == "GN":
                assert (self.gamma_GN_r is not None), "gamma_GN_r missing for GN row prior"

            if self.prior_c == "DP":
                assert (self.alpha_PY_c is not None), "alpha_PY_c missing for DP column prior"
            if self.prior_c == "PY":
                assert (self.alpha_PY_c is not None), "alpha_PY_c missing for PY column prior"
                assert (self.sigma_PY_c is not None), "sigma_PY_c missing for PY column prior"
                #Potential check for sigma outside of range
            if self.prior_c == "DM":
                assert (self.beta_DM_c is not None), "beta_DM_c missing for DM column prior"
                assert (self.K_star_DM_c is not None), "K_star_c missing for DM column prior"
            if self.prior_c == "GN":
                assert (self.gamma_GN_c is not None), "gamma_GN_c missing for GN column prior"

    def expected_cl(self, n, sigma, theta, H):
        n = int(n)
        if H == np.Infinity:
            if sigma == 0:
                output = theta * np.sum(1/(theta - 1 + np.array([_ for _ in range(1,n)])))
            else:
                output = 1/sigma*np.exp(gammaln(theta + sigma + n) - gammaln(theta + sigma) + gammaln(theta + 1)) - theta/sigma 
        elif H < np.Infinity:
            if sigma == 0:
                idx = np.array([_ for _ in range(n-1)])
                output = H - H*np.exp(np.sum(np.log(idx + theta*(1 - 1/H)) - np.log(theta + idx)))
        return output
    
    def expected_cl_gn(self):
            # np.exp()
        pass
