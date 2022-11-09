import numpy as np
import igraph as ig
from irm_undirected import irm
from irm_directed import irm_directed
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

def retrieve_samples(Z, gap = 25, burn_in = True):
    if burn_in == True:
        burn_in = len(Z)//2   
    return Z[burn_in::gap]

def cluster_summs(Z, ret = False):
    temp_Z = []
    mean_lenght = 0
    mean_nodes = 0
    for i in range(len(Z)):
        current_z = np.sum(Z[i], 0)
        temp_Z.append(current_z)
        mean_lenght += len(current_z)
        mean_nodes += np.mean(current_z)

    mean_lenght /= len(Z)
    mean_nodes /= len(Z)

    print('mean number of clusters', mean_lenght)
    print('mean nodes per cluster', mean_nodes)
    if ret:
        return temp_Z

# X = np.array([[0, 1, 1],
#               [1, 0, 0],
#               [1, 0, 0]])

# Z1 = np.array([[0, 1],
#                 [1, 0],
#                 [1, 0]])


# X = np.array([[0, 1, 1, 1, 1],
#               [1, 0, 1, 0, 0],
#               [1, 1, 0, 1, 0],
#               [1, 0, 1, 0, 0],
#               [1, 0, 0, 0, 0]])

# Z1 = np.array([[0, 1, 0],
#                 [1, 0, 0],
#                 [1, 0, 0],
#                 [1, 0, 0],
#                 [0, 0, 1]])

# A = np.array([np.where(Z1[i,:] == 1)[0] for i in range(len(Z1))]).flatten()

# Z1.T @ X @ Z1
# np.diag(np.sum(X@Z1*Z1, 0) / 2) 

# M1 = Z1.T @ X @ Z1 - np.diag(np.sum(X@Z1*Z1, 0) / 2) 

# m = np.sum(Z1, 0)[np.newaxis]
# # M = np.tile(m, (K, 1))

# M0 = m.T@m - np.diag((m*(m+1) / 2).flatten()) - M1 

# rhos = np.zeros((len(X), len(X)))

# for i in range(len(X)):
#     for j in range(len(X)):
#         if i == j:
#             continue
#         links = M1[A[i], A[j]]
#         non_links = M0[A[i], A[j]]
#         rhos[i,j] += (links + a) / (links + non_links + a + b)

def compute_rhos(X, Z):
    A = np.array([np.where(Z[i,:] == 1)[0] for i in range(len(Z))]).flatten()

    Z.T @ X @ Z
    np.diag(np.sum(X@Z*Z, 0) / 2) 

    M1 = Z.T @ X @ Z - np.diag(np.sum(X@Z*Z, 0) / 2) 

    m = np.sum(Z, 0)[np.newaxis]

    M0 = m.T@m - np.diag((m*(m+1) / 2).flatten()) - M1 

    rhos = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue
            links = M1[A[i], A[j]]
            non_links = M0[A[i], A[j]]
            rhos[i,j] += (links + a) / (links + non_links + a + b)

    return rhos

def compute_rho(X, sample):
    rhos = np.zeros((len(X), len(X)))
    for i in sample:
        rhos += compute_rhos(X, i)
    rhos /= len(sample)
    return rhos


g = ig.Graph.Read_GML('karate.txt')
X = np.array(g.get_adjacency().data)

T = 500
a = 1
b = 1
A = 5
Z = irm(X, T, a, b, A)

sample = retrieve_samples(Z)
cluster_summs(sample)

Z1 = Z[-1]


W = np.zeros((len(X),len(X))).astype(int)
W[0,1] = 1
W[1,0] = 1

def create_W(X, proportion = 0.1, symmetric = True, seed = 42, ret_indices = True):
    np.random.seed(seed)
    if symmetric:
        trX = X[np.triu_indices(len(X), k=1)]
        mask = trX>0
        
        
        links = len(trX[trX>0])
        nonlinks = len(trX[trX==0])


        draw = np.random.choice(links, size=round(links*proportion), replace = False)
        a1 = trX[mask] 
        a1[draw] = 0
        trX[mask] = a1

        
        draw = np.random.choice(nonlinks, size=round(nonlinks*proportion), replace = False)
        a2 = trX[~mask]
        a2[draw] = 1
        trX[~mask] = a2

        out = np.zeros((len(X), len(X)))
        out[np.triu_indices(out.shape[0], k=1)] = trX
        out = out + out.T

    if ret_indices:
        idxmatrix = W != X
        return out, idxmatrix.astype(int)
    else:
        return out

X_missing, W = create_W(X)

# X_missing = X-X*W
Z_missing = irm(X_missing, T, a, b, 5)
sample_missing = retrieve_samples(Z_missing)
cluster_summs(sample_missing)

rho = compute_rho(X, sample)
rho_missing = compute_rho(X, sample_missing)

def draw_roc(X, rhos):
    draws = np.random.binomial(1, rhos)
    pred = draws[np.triu_indices(len(draws), k=1)].flatten()
    fpr, tpr, _ = roc_curve(X[np.triu_indices(len(X), k=1)].flatten(), pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (AUC = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.show()

pred = np.random.binomial(1, rho).flatten()
fpr, tpr, _ = roc_curve(X.flatten(), pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()


W*rho_missing