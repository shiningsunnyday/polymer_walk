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
    mols = load_mols(args.motifs_folder)    
    red_grps = annotate_extra(mols, args.extra_label_path)
    # sanity checks
    G = nx.MultiDiGraph()
    # for i in range(41):
    #     G.add_node(name_group(i+1))
    pargs = [(i+1, j+1) for i in range(len(mols)) for j in range(len(mols))]        

    name_to_id = {name_group(i+1):i+1 for i in range(len(mols))}

    # a bunch of asserts
    # checks = []
    # checks.append('L3 S32'.split(' '))
    # checks.append('P15 P18'.split(' '))
    # checks.append('L4 P40'.split(' '))
    # checks.append('L4 P17'.split(' '))
    # checks.append('L4 P17'.split(' '))
    # checks.append('L4 P40'.split(' '))
    # checks.append('L4 P40'.split(' '))
    # checks.append('L4 S3'.split(' '))
    # checks.append('S2 S21'.split(' '))
    # checks.append('P15 S20'.split(' '))
    # checks.append('P15 S20'.split(' '))
    # checks.append('P15 S20'.split(' '))
    # checks.append('P15 S20'.split(' '))
    # checks.append('S32 L11'.split(' '))
    # checks.append('L3 S32'.split(' '))
    # checks.append('S7 S2'.split(' '))
    # checks.append('S21 L15'.split(' '))
    # checks.append('S5 S24'.split(' '))
    # checks.append('L3 S6'.split(' '))
    # checks.append('L1 S6'.split(' '))
    # checks.append('L11 S6'.split(' '))
    # checks.append('S16 S20'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('L16 S2'.split(' '))
    # checks.append('L16 S2'.split(' '))
    # checks.append('L16 S2'.split(' '))
    # checks.append('S10 L3'.split(' '))
    # checks.append('S10 L3'.split(' '))
    # checks.append('S10 L3'.split(' '))
    # checks.append('L16 S14'.split(' '))
    # checks.append('L16 L15'.split(' '))
    # checks.append('L16 S24'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('L16 S25'.split(' '))
    # checks.append('L21 S20'.split(' '))
    # checks.append('L21 P11'.split(' '))
    # checks.append('L21 S21'.split(' '))
    # checks.append('L21 P3'.split(' '))
    # checks.append('L21 P3'.split(' '))
    # checks.append('L21 P15'.split(' '))
    # checks.append('L21 S8'.split(' '))
    # checks.append('S10 L1'.split(' '))
    # checks.append('L4 P40'.split(' '))
    # checks.append('L4 P40'.split(' '))
    # checks.append('L4 P40'.split(' '))
    # checks.append('L4 P8'.split(' '))
    # checks.append('L4 P8'.split(' '))
    # checks.append('L4 P8'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P14'.split(' '))
    # checks.append('P22 P6'.split(' '))
    # checks.append('P22 P14'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P6'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P6'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 S2'.split(' '))
    # checks.append('P16 P11'.split(' '))
    # checks.append('S10 L4'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('S10 S21'.split(' '))
    # checks.append('S10 P15'.split(' '))
    # checks.append('S10 P15'.split(' '))
    # checks.append('S10 P15'.split(' '))
    # checks.append('S10 P15'.split(' '))
    # checks.append('L3 S32'.split(' '))
    # checks.append('L3 S32'.split(' '))
    # checks.append('L3 S32'.split(' '))
    # checks.append('L3 S32'.split(' '))
    # checks.append('L3 S32'.split(' '))
    # checks.append('L3 S32'.split(' '))
    # checks.append('L3 S32'.split(' '))
    # checks.append('L3 S32'.split(' '))
    # checks.append('L3 P28'.split(' '))
    # checks.append('L3 P28'.split(' '))
    # checks.append('S21 P15'.split(' '))
    # checks.append('L21 S21'.split(' '))
    # checks.append('L4 P28'.split(' '))
    # checks.append('L4 P28'.split(' '))
    # checks.append('L20 P15'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('P22 P9'.split(' '))
    # checks.append('S10 L3'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('L16 S2'.split(' '))
    # checks.append('L16 S32'.split(' '))
    # checks.append('P15 P9'.split(' '))
    # checks.append('S21 P40'.split(' '))
    # checks.append('P15 L18'.split(' '))
    # checks.append('P15 P6'.split(' '))
    # checks.append('L12 P17'.split(' '))
    # checks.append('S15 L4'.split(' '))
    # checks.append('S15 L4'.split(' '))
    # checks.append('P15 S32'.split(' '))
    # checks.append('P15 S2'.split(' '))
    # checks.append('P15 S32'.split(' '))
    # for check in checks:
    #     if not reds_isomorphic(*[name_to_id[c] for c in check]):
    #         if 'P40' not in check:
    #             breakpoint()
    # breakpoint()

    with Pool(32) as p:    
        res = p.starmap(reds_isomorphic, pargs)  
    # res = []
    # for arg in pargs:
    #     res.append(reds_isomorphic(*arg))  

    for (a, b), k in zip(pargs, res):
        if not k: continue
        max_size = max([len(key[0]+key[1]) for key in k])
        k = [key for key in k if len(key[0]+key[1])==max_size]
        for i, key in enumerate(k):
            r_grp_1, b1, b2, r_grp_2 = key
            G.add_edge(name_group(a), name_group(b), key=i, r_grp_1=r_grp_1, b1=b1, b2=b2, r_grp_2=r_grp_2)


    
    nx.write_adjlist(G, args.out_path)
    nx.write_edgelist(G, args.out_path.replace('adjlist', 'edgelist'), data=True)
    breakpoint()
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
