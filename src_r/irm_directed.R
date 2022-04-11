library(igraph)
library(pracma)

# setwd("C:/Users/nakaz/Desktop/DSBA/TESI/BNP-Net/src_r")
g<- read.graph("celegansneural.gml", format=c("gml"))
g <- read.graph("karate.txt", format=c("gml"))

gibbs_Sweep <- function(n){
  if (dim(z)[2] > 1){
    if (0 %in% colSums(z[-n,])){
      idx_empty <- which(colSums(z[-n,]) == 0)
      z <- z[, -idx_empty]
    }
    if (ncol(z) == 1){
      z <- matrix(1, N, 1)
    }
  }
  X_ <- X[-n , -n]
  # K = n.of components
  K <- ncol(z)
  
  if (K>1){
    m = colSums(z[-n,])
  } else {
    m = sum(z[-n])
  }
  M <- repmat(m, K, 1)
  dim(m) <- c(1, length(m))
  
  M1 <- t(z[-n,]) %*% X_ %*% z[-n,] - diag(colSums(X_ %*% z[-n,] * z[-n,]) / 2, nrow=K)
  
  if (K==1){
    M0 <- m*(m-1)/2 - M1
  } else{
    M0 <- t(m) %*% m - diag(as.vector(m*(m+1) / 2)) - M1
  }
  
  r <- t(t(z[-n,]) %*% X[-n, n])
  R <- repmat(r, K, 1)
  
  logLik_n <- lbeta(M1+R+a, M0+M-R+b) - lbeta(M1+a, M0+b)
  logLik_newcomp <- lbeta(r+a, m-r+b) - lbeta(a,b)
  
  logLikelihood <- rowSums(rbind(logLik_n, logLik_newcomp))
  logPrior <- log(c(m,A))
  
  logPosterior <- logPrior + logLikelihood
  
  P = exp(logPosterior - max(logPosterior))
  
  draw = runif(1)
  i = which(draw < cumsum(P)/sum(P))[1]
  
  z[n,] <- 0
  if (i == K+1){
    z <- cbind(z, rep(0,N))
  }
  
  z[n,i] <- 1
  
  if (0 %in% colSums(z)){
    idx_empty <- which(colSums(z) == 0)
    z <- z[, -idx_empty]
  }
  
  return (z)
}

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
      
      # X_ is adjacency matrix without node n
      X_ <- X[-n , -n]
      # K = n.of components
      K <- ncol(z)
      
      # m = n. of nodes in each component
      if (K>1){
        m = colSums(z[-n,])
        M <- repmat(m, K, 1) + diag(m) #CHECK
      } else {
        m = sum(z[-n])
        M <- repmat(m, K, 1) + m
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
      nonL2 <- t(M0)
      diag(M0_2) <- NA
      M0_2 <- t(matrix(t(M0_2)[which(!is.na(M0_2))], nrow=dim(M0_2)[1]-1, ncol=dim(M0_2)[2]))
      
      nonL <- cbind(M0, M0_2)
      
      # r = n. of links from current node to components
      r <- t(t(z[-n,]) %*% X[-n, n])
      R <- repmat(r, K, 1)
      
      # s = n. of links from components to current node
      s <- t(z[-n,]) %*% X[n, -n]
      S <- repmat(s, 1, K)
      
      
      
      L_n = matrix(0, nrow=dim(L)[1], ncol=dim(L)[2])
      # current_node_links + R
      L_n[1:dim(R)[1], 1:dim(R)[2]] <- L_n[1:dim(R)[1], 1:dim(R)[2]]+R
      
      #s_diag <- s #diag
      s_diag <- diag(as.vector(s))
      diag(L_n) <- diag(L_n) + as.vector(s)
      
      #ROBA SU S
      diag(S) <- NA #CHECKKKK
      S <- t(matrix(S[which(!is.na(S))], nrow=dim(S)[1]-1, ncol=dim(S)[2]))
      if (K>1){
        L_n[,(dim(L_n)[2] - dim(S)[2] + 1): dim(L_n)[2]] <- L_n[,(dim(L_n)[2] - dim(S)[2] + 1): dim(L_n)[2]] + S
      }
      
      
      
      M__2 <- M
      diag(M__2) <- NA
      M__2 <- t(matrix(t(M__2)[which(!is.na(M__2))], nrow=dim(M__2)[1]-1, ncol=dim(M__2)[2]))
      
      maxL_n <- cbind(M, M__2)
      
      
      logLik_old <- rowSums(lbeta(L+L_n+a, nonL+(maxL_n-L_n)+b) - lbeta(L+a, nonL+b))
      logLik_new <- sum(lbeta(c(r,s)+a, c(m,m)-c(r,s)+b) - lbeta(a,b))
      
      logLikelihood <- c(logLik_old, logLik_new)
      logPrior <- log(c(m,A))
      
      logPosterior <- logPrior + logLikelihood
      
      P = exp(logPosterior - max(logPosterior))
      
      draw = runif(1)
      i = which(draw < cumsum(P)/sum(P))[1]
      
      z[n,] <- 0
      if (i == K+1){
        z <- cbind(z, rep(0,N))
      }
      
      z[n,i] <- 1
      
      if (0 %in% colSums(z)){
        idx_empty <- which(colSums(z) == 0)
        z <- z[, -idx_empty]
      }
    }
    
    Z[[t]] <- z #save partition at the end of gibbs cycle
  }

Z <- irm(X,T,a,b,A)


X <- as_adjacency_matrix(g)
X <- matrix(X, nrow(X), ncol(X))
X[X>1] <- 1
X <- X[1:5, 1:5]

X <- t(matrix(c(0,1,1,0,1,
              1,0,0,0,1,
              1,0,0,1,1,
              1,1,1,0,0,
              0,0,1,1,0),nrow=5, ncol=5))

a <- 1
b <- 1
A <- 20
T <- 500

Z <- list()
N <- nrow(X)
z <- matrix(1, N, 1)

# n <- 3


