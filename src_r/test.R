library(igraph)



g <- read.graph("celegansneural.gml", format=c("gml"))
test <- read.graph("test.gml", format=c("gml"))
mat <- as_adjacency_matrix(test)
g_mat <- as_adjacency_matrix(g)
g_mat == 1

apply(mat,2,sum)==0

