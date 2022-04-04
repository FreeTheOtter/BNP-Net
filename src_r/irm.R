library(igraph)
library(pracma)


g <- read.graph("celegansneural.gml", format=c("gml"))
g <- read.graph("karate.txt", format=c("gml"))

X <- as_adjacency_matrix(g)
X <- matrix(X, nrow(X), ncol(X))
X <- X[1:5,1:5]

a <- 1
b <- 1
A <- 10

N = nrow(X)
#Initialize assignment vector (1 component)
z <- matrix(1, N, 1)
T <- 10
Z <- list()


n = 2
X_ <- X[-n , -n]
# K = n.of components
K <- ncol(z)

if (K>1){m = colSums(z[-n,])} else {m = sum(z[-n])}
M <- repmat(m, K, 1)
dim(m) <- c(1,length(m))

M1 <- t(z[-n,]) %*% X_ %*% z[-n,] - diag(colSums(X_ %*% z[-n,] * z[-n,]) / 2, nrow=K)

M0 <- t(m) %*% m - diag(m*(m+1) / 2, nrow=K) - M1

#TODO: fix Error in diag(m * (m + 1)/2, nrow = K) : 
#'nrow' or 'ncol' cannot be specified when 'x' is a matrix


r <- z[-n,] %*% X[-n, n]
R <- repmat(r, K, 1)

logLik_n <- lbeta(M1+R+a, M0+M-R+b) - lbeta(M1+a, M0+b)
logLik_newcomp <- lbeta(r+a, m-r+b) - lbeta(a,b)

logLikelihood <- rowSums(rbind(logLik_n, logLik_newcomp))
logPrior <- log(c(m,A))

logPosterior <- logPrior + logLikelihood

P = exp(logPosterior - max(logPosterior))

draw = runif(1)
i = which(draw < cumsum(P)/sum(P))

z[n,] <- 0
if (i == K+1){
  z <- cbind(z, rep(0,N))
}

z[n,i] <- 1

idx_empty <- which(colSums(z) == 0)
z <- z[, -idx_empty]

Z[[1]] <- z
