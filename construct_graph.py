import networkx as nx
import argparse
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from utils import *
from tqdm import tqdm
import re


def search_walk(i, walk, graph, cur):
    if i == len(walk)-1: return cur
    breakpoint()
    a = walk[i]
    b = walk[i+1]
    for e in graph[a][b]:
        cur.append(e)
        new_cur = search_walk(i+1, walk, graph, cur)
        if new_cur: return new_cur
        cur.pop()
    return []


def verify_walk(r_lookup, graph, walk):
    # r_lookup: dict(red group id: atoms)
    # check all edges exist
    for i in range(len(walk)):
        walk[i], walk[i+1]
    
    cur = []
    return search_walk(0, walk, graph, cur)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motifs_folder')
    parser.add_argument('--extra_label_path')    
    parser.add_argument('--data_file')
    parser.add_argument('--predefined_graph_file')
    parser.add_argument('--graph_vis_file')
    args = parser.parse_args()
    mols = load_mols(args.motifs_folder)
    red_grps = annotate_extra(mols, args.extra_label_path)    
    graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
    lines = open(args.data_file).readlines()    
    e = []
    nodes = []
    breakpoint()
    r_lookup = r_member_lookup(mols)
    for i, l in enumerate(lines):        
        walk = l.rstrip('\n')        
        grp = '(?:[SPL][0-9]+)'
        pat_inner = f'\[->{grp}(?:,->{grp})*\]'
        pat = f'(\w+)(?:->((\w+{pat_inner})|(\w+)))*'
        g = re.fullmatch(pat, walk)
        if g:
            path = verify_walk(r_lookup, graph, g.groups())
        else:
            breakpoint()
        if not path: continue
        for a, b in zip(walk[:-1], walk[1:]):
            e.append([a, b])
            e.append([b, a])
            try:
                for j in graph[a][b]:
                    graph[a][b][j]['weight'] = graph[a][b][j].get('weight', 0) + 2
                for j in graph[b][a]:
                    graph[b][a][j]['weight'] = graph[b][a][j].get('weight', 0) + 2
            except:
                continue
    print(nodes)
    # G = nx.MultiDiGraph(e)
    fig = plt.Figure(figsize=(100, 100))
    ax = fig.add_subplot()
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, with_labels=True, ax=ax)
    for edge in tqdm(graph.edges(data='weight')):
        if edge[2] and edge[2] >= 2:
            nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=edge[2], ax=ax)
    fig.savefig(args.graph_vis_file)
