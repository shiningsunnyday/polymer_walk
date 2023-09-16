import networkx as nx
import argparse
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from utils import *
from tqdm import tqdm
import pickle
import re
import matplotlib.colors as mcolors

base_colors = list(mcolors.BASE_COLORS)
grp = '|'.join([f'(?:{name_group(i)})' for i in range(1, 98)])
grp = f'(?:{grp})'
pat_chain = f'(?:->{grp})+'
pat_inner = f'\[(?:(?:,?){pat_chain})+\]'
pat_or = f'(?:({grp}{pat_inner})|({grp}))'
pat = f'(?:{pat_or})(?:->{pat_or})*'

def extract_chain(walk):
    side_chains = []
    while True:
        chain_search = re.search(pat_inner, walk)
        if not chain_search: break
        start, end = chain_search.span()
        side_chains.append(walk[start:end])
        walk = walk[:start]+'#'+walk[end:] 
    
    res = walk.split('->')
    j = 0
    for i in range(len(res)):
        if '#' in res[i]:
            res[i] = res[i].replace('#', side_chains[j])
            j += 1
    return res    

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


def dfs_traverse(walk):
    prev_node = None
    root_node = None
    conn = []
    id = 0
    for i in range(len(walk)):
        if '[' in walk[i]:
            start = walk[i].find('[')
            assert ']' == walk[i][-1]
            grps = [walk[i][:start]]
            for grp in walk[i][start+1:-1].split(','):
                grps.append(grp[2:])
            cur = Node(prev_node, [], grps[0], id)
            id += 1
            if i: 
                conn.append((prev_node, cur))
                conn.append((cur, prev_node))
            for g in grps[1:]:
                g_child = Node(cur, [], g, id, True)
                id += 1
                cur.add_child(g_child)
                conn.append((cur, g_child))
                conn.append((g_child, cur))

            if i:
                prev_node.add_child(cur)           
            prev_node = cur
        else:
            cur = Node(prev_node, [], walk[i], id)
            id += 1
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
        raise
    used_reds = defaultdict(set)
    edge_conn = []
    bad = False
    for a, b in conn:
        for i in range(len(graph[a.val][b.val])):            
            e = graph[a.val][b.val][i]
            red_j1 = find_edge(e, 'k1', r_lookup[a.val])
            if set(red_j1) & used_reds[a]: 
                if i == len(graph[a.val][b.val])-1: bad = True
                continue
            else: 
                used_reds[a] |= set(red_j1)
                # print(f"{a.val}->{b.val}")
                # print(f"used {used_reds[a]}")
                edge_conn.append((a, b, i))
                if b in a.children:
                    ind = a.children.index(b)
                    a.children[ind] = (b, i)
                elif b == a.parent:
                    a.parent = (b, i)
                break
        if bad: raise
     
    print("done")    
    return root, edge_conn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motifs_folder')
    parser.add_argument('--extra_label_path')    
    parser.add_argument('--data_file')
    parser.add_argument('--predefined_graph_file')
    parser.add_argument('--graph_vis_file')
    parser.add_argument('--out_path', required=False)
    args = parser.parse_args()
    mols = load_mols(args.motifs_folder)
    red_grps = annotate_extra(mols, args.extra_label_path)    
    graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
    lines = open(args.data_file).readlines()    
    r_lookup = r_member_lookup(mols)
    
    dags = {}
    for i, l in enumerate(lines):        
        walk = l.rstrip('\n')      
        grps = extract_chain(walk) 
        print(f"walk: {walk}, grps: {grps}")
        try:             
            root, conn = verify_walk(r_lookup, graph, grps)
            dags[i] = (grps, root, conn)
        except Exception as e:
            print("verify failed")
            continue
        for a, b, e in conn:
            try:
                graph[a.val][b.val][e]['weight'] = graph[a.val][b.val][e].get('weight', 0)+2
                graph[b.val][a.val][e]['weight'] = graph[b.val][a.val][e].get('weight', 0)+2
                graph[a.val][b.val][e]['color'] = graph[b.val][a.val][e]['color'] = i
            except:
                continue                

    if args.out_path: 
        print(f"processed {len(dags)} dags")
        pickle.dump(dags, open(args.out_path, 'wb+'))
        breakpoint()

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
