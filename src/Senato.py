import numpy as np
import pandas as pd
import json
from main_sbm import SBM
import matplotlib.pyplot as plt
import seaborn as sns
import igraph as ig
import pickle as pkl
from collections import Counter
with open("datasets/edges_se2013.json") as f:
    df = pd.DataFrame(json.load(f))

with open("datasets/senators_2013.json") as f:
    senators = pd.DataFrame(json.load(f))


set1 = set(df['i'].unique())
set2 = set(df['j'].unique())

senatori = list(set.union(set1, set2))

sen_dict = {}
# n = [_ for _ in range(len(senatori))]
for i in range(len(senatori)):
    sen_dict[senatori[i]] = int(i) 

df2 = df.copy()
for i in range(len(df)):
    df2.iloc[i,0] = sen_dict[df.iloc[i,0]]
    df2.iloc[i,1] = sen_dict[df.iloc[i,1]]

X = np.zeros((len(senatori), len(senatori)))

for i in range(len(df)):
    X[df2.iloc[i,0], df2.iloc[i,1]] = df2.iloc[i,2]

config_uni = {'directed': True,
          'binary': True,
          'unicluster': True}

config_bi = {'directed': True,
          'binary': True,
          'unicluster': False}

np.where(np.sum(X, axis=0) == 0)



fig, ax = plt.subplots(figsize = (10,10))
ax = sns.heatmap(X, linewidth=0.0, ax = ax, cbar=True, cmap= 'rocket_r', xticklabels=[], yticklabels=[])
plt.show()

X2 = np.zeros((len(senatori), len(senatori)))
X2[X>0] = 1
fig, ax = plt.subplots(figsize = (10,10))
ax = sns.heatmap(X2, linewidth=0.0, ax = ax, cmap= 'rocket_r', xticklabels=[], yticklabels=[])
plt.show()

X3 = X.copy()
X3[X>10] = 10
fig, ax = plt.subplots(figsize = (10,10))
ax = sns.heatmap(X3, linewidth=0.0, ax = ax, cbar=True, cmap= 'rocket_r', xticklabels=[], yticklabels=[])
plt.show()

PY = SBM(config_uni, prior_r = "PY", alpha_PY_r=0.6,sigma_PY_r=-0.3, 
                start_z = "singleton", set_seed=42)
PY.fit(X2,100)
PY.compute_block_probabilities()
PY.predict()

for i in PY.Z:
     print(PY.evalLogLikelihood_full(i))

GN = SBM(config_uni, prior_r = "GN", gamma_GN_r=0.45,
            start_z = "singleton", set_seed=42)
GN.fit(X2,100)
GN.compute_block_probabilities()
GN.predict()

idx_list = np.array([])
for i in range(PY.z_hat.shape[1]):
     idx_list = np.append(idx_list, np.argwhere(PY.c_hat == i).flatten())
    #  print(len(np.argwhere(PY.c_hat == i)))

test = X2[np.ix_(idx_list.astype(int),idx_list.astype(int))]
fig, ax = plt.subplots(figsize = (10,10))
ax = sns.heatmap(test, linewidth=0.0, ax = ax,cbar=False, cmap= 'rocket_r', xticklabels=[], yticklabels=[])
plt.show()

PY_bi = SBM(config_bi, prior_r = "PY", alpha_PY_r=0.6,sigma_PY_r=-0.3, 
         prior_c = "PY", alpha_PY_c=0.6,sigma_PY_c=-0.3, 
                start_z = "singleton", set_seed=42)
PY_bi.fit(X2,200)
PY_bi.compute_block_probabilities()
PY_bi.predict()

for i, j in zip(PY_bi.Zr, PY_bi.Zc):
     print(PY.evalLogLikelihood_full(i,j))

idx_list_rows = np.array([])
for i in range(PY_bi.zr_hat.shape[1]):
     idx_list_rows = np.append(idx_list_rows, np.argwhere(PY_bi.cr_hat == i).flatten())
    #  print(len(np.argwhere(PY.c_hat == i)))

idx_list_columns = np.array([])
for i in range(PY_bi.zc_hat.shape[1]):
     idx_list_columns = np.append(idx_list_columns, np.argwhere(PY_bi.cc_hat == i).flatten())

test_bi = X2[np.ix_(idx_list_rows.astype(int),idx_list_columns.astype(int))]
fig, ax = plt.subplots(figsize = (10,10))
ax = sns.heatmap(test_bi, linewidth=0.0, ax = ax, cbar = False, cmap= 'rocket_r', xticklabels=[], yticklabels=[])
plt.show()

for i in range(PY_bi.zc_hat.shape[1]):
    print(len(np.argwhere(PY.cr_hat == i)))
    #  idx_list_rows = np.append(idx_list, np.argwhere(PY_bi.cr_hat == i).flatten())
from collections import Counter
Counter(PY_bi.cr_hat)
Counter(PY_bi.cc_hat)

