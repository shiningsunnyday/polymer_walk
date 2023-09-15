import argparse
import os
from itertools import product, permutations
import networkx as nx
from multiprocessing import Pool
import matplotlib.pyplot as plt
from copy import deepcopy
from utils import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra_label_path')
    parser.add_argument('--out_path')
    parser.add_argument('--motifs_folder')
    args = parser.parse_args()
    red_grps = annotate_extra(mols, args.extra_label_path)
    # sanity checks
    G = nx.MultiDiGraph()
    # for i in range(41):
    #     G.add_node(name_group(i+1))
    pargs = [(i+1, j+1) for i in range(97) for j in range(97)]        
    with Pool(32) as p:    
        res = p.starmap(reds_isomorphic, pargs)  
    # res = []
    # for arg in pargs:
    #     res.append(reds_isomorphic(*arg))  

    for (a, b), k in zip(pargs, res):
        if not k: continue
        for i, key in enumerate(k):
            k1, k2 = key
            G.add_edge(name_group(a), name_group(b), key=i, k1=k1, k2=k2)


    
    nx.write_adjlist(G, args.out_path)
    nx.write_edgelist(G, args.out_path.replace('adjlist', 'edgelist'), data=True)
    fig = plt.Figure(figsize=(200, 200))
    ax = fig.add_subplot()
    pos = nx.circular_layout(G)
    for e in G.edges(data=True, keys=True):
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                    ),
                                    ),
                    )    
    nx.draw(G, pos=pos, with_labels=True, ax=ax)        
    fig.savefig(args.out_path.replace('.adjlist', '.png'))
    os.makedirs(os.path.join(args.motifs_folder, f'with_label/'), exist_ok=True)
    for i, mol in enumerate(mols):
        Draw.MolToFile(mol, os.path.join(args.motifs_folder, f'with_label/{i+1}.png'), size=(2000, 2000))
