import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from generation import generate_graph


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

def compute_rhos(X, Z, a=1, b=1, edge_type = 'undirected', edge_weight = 'binary', mode = 'normal'):    
    A = np.array([np.where(Z[i,:] == 1)[0] for i in range(len(Z))]).flatten()

    if edge_type == 'undirected':
        M1 = Z.T @ X @ Z - np.diag(np.sum(X@Z*Z, 0) / 2) 

        m = np.sum(Z, 0)[np.newaxis]

        M0 = m.T@m - np.diag((m*(m+1) / 2).flatten()) - M1 

    elif edge_type == 'directed':
        M1 = Z.T @ X @ Z

        m = np.sum(Z, 0)[np.newaxis]
        M0 = m.T@m - np.diag(m.flatten()) - M1

    rhos = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue
            links = M1[A[i], A[j]]
            non_links = M0[A[i], A[j]]
            rhos[i,j] += (links + a) / (links + non_links + a + b)
    return rhos

def compute_rho(X, sample, edge_type = 'undirected', edge_weight = 'binary', mode = 'normal'):
    rhos = np.zeros((len(X), len(X)))
    for i in sample:
        rhos += compute_rhos(X, i, edge_type = edge_type, edge_weight = edge_weight, mode = mode)
    
    rhos /= len(sample)
    return rhos

def create_W(X, prop_links = 0.1, prop_nonlinks = 0.1, symmetric = True, rand_ = False, seed = 42, ret_indices = True):
    if rand_ == False:
        np.random.seed(seed)
    
    if symmetric:
        trX = X[np.triu_indices(len(X), k=1)]
        mask = trX>0
        
        
        links = len(trX[trX>0])
        nonlinks = len(trX[trX==0])


        draw = np.random.choice(links, size=round(links*prop_links), replace = False)
        a1 = trX[mask] 
        a1[draw] = 0
        trX[mask] = a1

        
        draw = np.random.choice(nonlinks, size=round(nonlinks*prop_nonlinks), replace = False)
        a2 = trX[~mask]
        a2[draw] = 1
        trX[~mask] = a2

        out = np.zeros((len(X), len(X)))
        out[np.triu_indices(out.shape[0], k=1)] = trX
        out = out + out.T

    if ret_indices:
        idxmatrix = out != X
        return out, idxmatrix.astype(int)
    else:
        return out

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
