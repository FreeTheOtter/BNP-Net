library(igraph)
library(pracma)

g <- read.graph("karate.txt", format=c("gml"))

irm <- function(X, T, a, b, A){
  Z <- list() #Initialize empty list to hold partition assignment each gibbs sweep
  N <- nrow(X)
  z <- matrix(1, N, 1) #Initialize assignment vector (1 component)
  
  for (t in 1:T){# for each iteration t
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
      if (K>1){
        m = colSums(z[-n,])
      } else {
        m = sum(z[-n])
      }
      # M = max link matrix
      M <- repmat(m, K, 1)
      
      dim(m) <- c(1, length(m)) #upgrade to matrix to be able to transpose later
      
      # L = links matrix between components without current node
      L <- t(z[-n,]) %*% X_ %*% z[-n,] - diag(colSums(X_ %*% z[-n,] * z[-n,]) / 2, nrow=K)
      
      # nonL = non-links matrix between components without current node
      if (K==1){
        nonL <- m*(m-1)/2 - L
      } else{
        nonL <- t(m) %*% m - diag(as.vector(m*(m+1) / 2)) - L
      }
      
      # r = n. of links from current node to components
      r <- t(t(z[-n,]) %*% X[-n, n])
      R <- repmat(r, K, 1)
      
      logLik_old <- lbeta(L+R+a, nonL+(M-R)+b) - lbeta(L+a, nonL+b)
      logLik_new <- lbeta(r+a, m-r+b) - lbeta(a,b)
      
      logLikelihood <- rowSums(rbind(logLik_old, logLik_new))
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
      
    }
    Z[[t]] <- z #save partition at the end of gibbs cycle
  }
  return (Z)
}


X <- as_adjacency_matrix(g)
X <- matrix(X, nrow(X), ncol(X))

a <- 1
b <- 1
A <- 10
T <- 500
set.seed(6)
Z <- irm(X,T,a,b,A)

for (i in 0:10){
  print(colSums(Z[[length(Z) - i]]))
}


