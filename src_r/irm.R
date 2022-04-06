library(igraph)
library(pracma)

#setwd("C:/Users/nakaz/Desktop/DSBA/TESI/BNP-Net/src_r")
g <- read.graph("celegansneural.gml", format=c("gml"))
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

irm <- function(X, T, a, b, A){
  for (t in 1:T){
    for (n in 1:N){
      z <- gibbs_Sweep(n)
    }
    Z[[t]] <- z #save partition at the end of gibbs cycle
  }
}


X <- as_adjacency_matrix(g)
X <- matrix(X, nrow(X), ncol(X))

a <- 1
b <- 1
A <- 10

N = nrow(X)
#Initialize assignment vector (1 component)
z <- matrix(1, N, 1)
T <- 10
Z <- list()

n <- 3
z <- gibbs_Sweep(n)

Z[[n]] <- z