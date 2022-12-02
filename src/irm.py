import numpy as np
import igraph as ig
from irm_directed import irm_directed
from irm_undirected import irm_undirected
from irm_directed_separate import irm_directed_separate
from irm_weighted import irm_directed_weighted

def irm(X, T, a, b, A, set_seed = False, random_seed = 42, print_iter = False,
         edge_type = 'directed', edge_weight = 'binary', mode = 'normal'):
    
    if edge_type == 'undirected':
        if edge_weight == 'binary':
            Z = irm_undirected(X, T, a, b, A , set_seed, random_seed, print_iter)
        elif edge_weight == 'integer':
            pass
            #todo
    
    elif edge_type == 'directed':
        if edge_weight == 'binary':
            if mode == 'normal':
                Z = irm_directed(X, T, a, b, A , set_seed, random_seed, print_iter)
            elif mode == 'separate': # in this case, Z is a list of [Z_out, Z_in]
                Z = irm_directed_separate(X, T, a, b, A , set_seed, random_seed, print_iter)
        
        elif edge_weight == 'integer':
            if mode == 'normal':
                Z = irm_directed_weighted(X, T, a, b, A , set_seed, random_seed, print_iter)

    
    return Z