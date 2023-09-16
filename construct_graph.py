import networkx as nx
import argparse
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from utils import *
from tqdm import tqdm
import re
import matplotlib.colors as mcolors

base_colors = list(mcolors.BASE_COLORS)
grp = '(?:[SPL][0-9]+)'
pat_chain = f'(?:->{grp})+'
pat_inner = f'\[{pat_chain}(?:,{pat_chain})*\]'
pat_or = f'({grp}{pat_inner})|({grp})'
pat = f'(?:{pat_or})(?:->({pat_or}))*'

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

    
class Node:
    def __init__(self, parent, children, val):        
        self.parent = parent
        self.children = children
        self.val = val

    def add_child(self, child):
        self.children.append(child)


def dfs_traverse(walk):
    prev_node = None
    root_node = None
    conn = []
    for i in range(len(walk)):
        if '[' in walk[i]:
            match = re.match(f'({grp})\[->({grp})(?:,->({grp}))*\]', walk[i])
            grps = match.groups()
            cur = Node(prev_node, [], grps[0])
            if i: 
                conn.append((prev_node, cur))
                conn.append((cur, prev_node))
            for g in grps[1:]:
                g_child = Node(cur, [], g)
                cur.add_child(g_child)
                conn.append((cur, g_child))
                conn.append((g_child, cur))

            if i:
                prev_node.add_child(cur)           
            prev_node = cur
        else:
            cur = Node(prev_node, [], walk[i])
            if i: 
                prev_node.add_child(cur)
                conn.append((prev_node, cur))
                conn.append((cur, prev_node))
            prev_node = cur

        root_node = root_node or prev_node

    conn.append((root_node, prev_node))
    conn.append((prev_node, root_node))

    return root_node, conn



def verify_walk(r_lookup, graph, walk):
    # r_lookup: dict(red group id: atoms)
    # check all edges exist
    def find_edge(red_key, dir, r_grps):
        for i, r_grp in r_grps.items():
            if dir == 'k1' and r_grp == red_key[dir][:len(r_grp)]:
                return r_grp
            if dir == 'k2' and r_grp == red_key[dir][-len(r_grp):]:
                return r_grp
        breakpoint()
    
    try:
        root, conn = dfs_traverse(walk)
    except:
        breakpoint()
    used_reds = defaultdict(set)
    edge_conn = []
    for a, b in conn:
        for i in range(len(graph[a.val][b.val])):            
            e = graph[a.val][b.val][i]
            red_j1 = find_edge(e, 'k1', r_lookup[a.val])
            if set(red_j1) & used_reds[a]: 
                continue
            else: 
                used_reds[a] |= set(red_j1)
                # print(f"{a.val}->{b.val}")
                # print(f"used {used_reds[a]}")
                edge_conn.append((a, b, i))
                break
     
    print("done")
    
    return root, edge_conn


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
    r_lookup = r_member_lookup(mols)
    
    for i, l in enumerate(lines):        
        walk = l.rstrip('\n')  
        g = re.fullmatch(pat, walk)
        if g:         
            grps = [grp for grp in g.groups() if grp]
            print(f"walk: {walk}, grps: {grps}")
            try:   
                root, conn = verify_walk(r_lookup, graph, grps)
            except:
                print("verify failed")
                continue
        else:
            breakpoint()
            print("regex failed")
            continue
        for a, b, e in conn:
            try:
                graph[a.val][b.val][e]['weight'] = graph[a.val][b.val][e].get('weight', 0)+2
                graph[b.val][a.val][e]['weight'] = graph[b.val][a.val][e].get('weight', 0)+2
                graph[a.val][b.val][e]['color'] = graph[b.val][a.val][e]['color'] = i
            except:
                continue
    # G = nx.MultiDiGraph(e)
    fig = plt.Figure(figsize=(100, 100))
    ax = fig.add_subplot()
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, with_labels=True, ax=ax)
    for edge in tqdm(graph.edges(data=True)):
        if 'weight' not in edge[2]: continue
        weight = edge[2]['weight']
        color = edge[2]['color']
        index = color % len(base_colors)
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=weight, ax=ax, edge_color=[base_colors[index]])
    fig.savefig(args.graph_vis_file)
    print(args.graph_vis_file)
