library(igraph)
library(pracma)

setwd("C:/Users/nakaz/Desktop/DSBA/TESI/BNP-Net/src_r")
g<- read.graph("celegansneural.gml", format=c("gml"))

irm_directed <- function(X, T, a, b, A){
  Z <- list() #Initialize empty list to hold partition assignment each gibbs sweep
  N <- nrow(X)
  z <- matrix(1, N, 1) #Initialize assignment vector (1 component)
  
  for (t in 1:T){# for each iteration t
    print(t)
    for (n in 1:N){#for each node n
      
      # If partition without node n has empty component, remove it
      if (dim(z)[2] > 1){
        if (0 %in% colSums(z[-n,])){
          idx_empty <- which(colSums(z[-n,]) == 0)
          z <- z[, -idx_empty]
        }
        if (ncol(z) == 1){
          z <- matrix(1, N, 1)
        }
      }
      
      # X_ = adjacency matrix without node n
      X_ <- X[-n , -n]
      # K = n.of components
      K <- ncol(z)
      
      # m = n. of nodes in each component
      # M1 = max link matrix
      if (K>1){
        m = colSums(z[-n,])
        M1 <- repmat(m, K, 1) + diag(m) 
      } else {
        m = sum(z[-n])
        M1 <- repmat(m, K, 1) + m
      }
      
      dim(m) <- c(1, length(m)) #upgrade to matrix to be able to transpose later
      
      
      
      # L1 = outgoing links matrix between components without current node
      L1 <- t(z[-n,]) %*% X_ %*% z[-n,]
      
      # L2 = incoming links matrix between components without current node
      # L2 is L1 without diagonal, and transposed
      L2 <- t(L1)
      diag(L2) <- NA
      L2 <- t(matrix(t(L2)[which(!is.na(L2))], nrow=dim(L2)[1]-1, ncol=dim(L2)[2]))
      
      # L = full matrix of links, obtained by attaching L1 and L2, obtaining a non-square matrix
      # This "escamotage" is needed since we need to consider each ORDERED pair of clusters to compute likelihood
      # More info on picture
      L <- cbind(L1,L2)
      
      
      
      # nonL1 = outgoing non-links between components without current node
      if (K==1){
        nonL1 <- t(m) %*% m - diag(m) - L1
      } else{
        nonL1 <- t(m) %*% m - diag(as.vector(m)) - L1
      }
      
      # nonL2 = incoming non-links between components without current node
      nonL2 <- t(nonL1)
      diag(nonL2) <- NA
      nonL2 <- t(matrix(t(nonL2)[which(!is.na(nonL2))], nrow=dim(nonL2)[1]-1, ncol=dim(nonL2)[2]))
      
      # nonL = full matrix of non links
      nonL <- cbind(nonL1, nonL2)
      
      
      
      # r = n. of outgoing links from current node to components
      r <- t(t(z[-n,]) %*% X[-n, n])
      R <- repmat(r, K, 1)
      
      # s = n. of incoming links from components to current node
      s <- t(z[-n,]) %*% X[n, -n]
      S <- repmat(s, 1, K)
      
      
      
      # L_n = currently sampled node links, initialized as a matrix of zeros
      L_n = matrix(0, nrow=dim(L)[1], ncol=dim(L)[2])
      # current node links + R (all outgoing links)
      L_n[1:dim(R)[1], 1:dim(R)[2]] <- L_n[1:dim(R)[1], 1:dim(R)[2]]+R
      
      # Adding to L_n diagonal incoming links (as there is a single prob. parameter within cluster)
      diag(L_n) <- diag(L_n) + as.vector(s)
      
      # Remove diagonal from S (incoming links to current node matrix) and transposing,
      # to sum to right side of link matrix
      diag(S) <- NA #CHECKKKK
      S <- t(matrix(S[which(!is.na(S))], nrow=dim(S)[1]-1, ncol=dim(S)[2]))
      if (K>1){
        L_n[,(dim(L_n)[2] - dim(S)[2] + 1): dim(L_n)[2]] <- L_n[,(dim(L_n)[2] - dim(S)[2] + 1): dim(L_n)[2]] + S
      }
      
      
      # Update M1 (Max link matrix) with incoming max links M2
      M2 <- M1
      diag(M2) <- NA
      M2 <- t(matrix(t(M2)[which(!is.na(M2))], nrow=dim(M2)[1]-1, ncol=dim(M2)[2]))
      
      maxL_n <- cbind(M1, M2)
      
      
      logLik_old <- rowSums(lbeta(L+L_n+a, nonL+(maxL_n-L_n)+b) - lbeta(L+a, nonL+b))
      logLik_new <- sum(lbeta(c(r,s)+a, c(m,m)-c(r,s)+b) - lbeta(a,b))
      
      logLikelihood <- c(logLik_old, logLik_new)
      logPrior <- log(c(m,A))
      
      logPosterior <- logPrior + logLikelihood
      
      # normalized probability vector
      P = exp(logPosterior - max(logPosterior))
      
      # Draw from uniform, assign to component i
      draw = runif(1)
      i = which(draw < cumsum(P)/sum(P))[1]
      
      z[n,] <- 0
      if (i == K+1){#if new component, add new column to partition assignment matrix z
        z <- cbind(z, rep(0,N))
      }
      z[n,i] <- 1
    }
    
    Z[[t]] <- z #save partition at the end of gibbs cycle
  }
  
  return (Z)
}




X <- as_adjacency_matrix(g)
X <- matrix(X, nrow(X), ncol(X))
X[X>1] <- 1 #There are some 2 values in the adjacency matrix for some reason (probably renaming from label to node number in the dataset) 

a <- 1
b <- 1
A <- 20
T <- 100
set.seed(42)
Z <- irm_directed(X,T,a,b,A)

for (i in 0:10){
  print(colSums(Z[[length(Z) - i]]))
}
