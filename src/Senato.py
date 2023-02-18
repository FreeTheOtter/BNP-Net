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