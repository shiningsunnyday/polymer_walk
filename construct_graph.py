import networkx as nx
import argparse
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from utils import *
from tqdm import tqdm
import pickle
import re
import json
import matplotlib.colors as mcolors

base_colors = list(mcolors.TABLEAU_COLORS.values())

test_walks = [
    ['L3-2>S32-2>S20[-1>S1,-4>S1]-2>S32-2>L3', 'L3->S32->S20[->S1,->S1]->S32->L3'],
    ['L3-2>S32-2>S20[-1>S1,-4>S1]-2>S32-2>L3', 'L3->S32->S20[->S1,->S1]->S32->L3']
]
def run_tests(r_lookup, graph):
    for walk, walk_with_e in test_walks:
        root_1, conn_1 = verify_walk(r_lookup, graph, chain_extract(walk, graph))
        root_2, conn_2 = verify_walk(r_lookup, graph, extract_chain(walk_with_e))
        assert set([(a.val, b.val) for (a, b, _) in conn_1]) == set((a.val, b.val) for (a, b, _) in conn_2)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motifs_folder')
    parser.add_argument('--extra_label_path')    
    parser.add_argument('--data_file')
    parser.add_argument('--predefined_graph_file')
    parser.add_argument('--trained_E_file')
    parser.add_argument('--graph_vis_file')
    parser.add_argument('--out_path', required=False)

    # some additional args
    parser.add_argument('--extract_edges', action='store_true')
    parser.add_argument('--save_edges_dir', help='where to save edges if --extract_edges')

    args = parser.parse_args()
    mols = load_mols(args.motifs_folder)
    red_grps = annotate_extra(mols, args.extra_label_path)    
    graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)    
    r_lookup = r_member_lookup(mols)
    if args.trained_E_file:
        E_trained = json.load(open(args.trained_E_file, 'r'))
    dags = {}
    used_edges = set()
    num_colors = 0
    roots = []
    conns = []
    if os.path.isfile(args.data_file):
        lines = open(args.data_file).readlines()
        # run some tests
        run_tests(r_lookup, graph)
        files = set()
        for i, l in enumerate(lines): 
            name = l.split()[0]
            l = l[l.find(' ')+1:]
            walk = l.rstrip('\n').split(' ')[0]
            if args.extract_edges: grps = chain_extract(walk, graph)                
            else: grps = extract_chain(walk)
            # print(f"walk: {walk}, grps: {grps}")
            try:
                root, conn = verify_walk(r_lookup, graph, grps)
                if args.extract_edges and args.save_edges_dir:
                    os.makedirs(args.save_edges_dir, exist_ok=True)
                    save_file = os.path.join(args.save_edges_dir, f'{name}.edgelist')
                    if save_file in files:
                        breakpoint()                    
                    nx.write_edgelist(grps, save_file, data=True)                
                    files.add(save_file)
            except (KeyError, RuntimeError) as e:
                breakpoint()
                if type(e) == KeyError:
                    print(e)
                else:
                    print(e)
                continue            
            if root.id:
                breakpoint()
            dags[i] = (grps, root, conn)           
            roots.append(root)
            conns.append(conn)
    else: # other dataset
        lines = sorted(os.listdir(args.data_file), key=lambda f: int(''.join(list(filter(lambda x: x.isdigit(), f)))))
        for f in lines:
            G = nx.read_edgelist(os.path.join(args.data_file, f), create_using=nx.MultiDiGraph)      
            if G.nodes() and max(len(cyc) for cyc in nx.simple_cycles(G)) > 2: # length > 2 is problem
                print(f"{f} has cycle")
            try:
                root, conn = bfs_traverse(G, graph)
            except (KeyError, RuntimeError) as e:
                breakpoint()
                root, conn = bfs_traverse(G, graph)
                if type(e) == KeyError:
                    print(e)
                else:
                    breakpoint()
                continue
            if root.id:
                breakpoint()   
            dags[int(f.split('walk_')[-1].split('.')[0])] = (None, root, conn)
            roots.append(root)
            conns.append(conn)


    for i, conn in enumerate(conns):
        do_color = False
        if num_colors >= 5: do_color = False
        else:
            for a, b, e in conn:
                if (a.val,b.val,e) in used_edges:
                    do_color = False
            num_colors += do_color
        for a, b, e in conn:            
            try:
                graph[a.val][b.val][e]['weight'] = graph[a.val][b.val][e].get('weight', 0)+.5
                graph[b.val][a.val][e]['weight'] = graph[b.val][a.val][e].get('weight', 0)+.5
                if do_color: 
                    graph[a.val][b.val][e]['color'] = graph[b.val][a.val][e]['color'] = base_colors[num_colors%len(base_colors)]
                    used_edges.add((a.val,b.val,e))
                    graph[a.val][b.val][e]['weight'] = E_trained[a.val][b.val]*100 if args.trained_E_file else 40
                    graph[b.val][a.val][e]['weight'] = E_trained[b.val][a.val]*100 if args.trained_E_file else 40
                else: 
                    graph[a.val][b.val][e]['color'] = graph[b.val][a.val][e]['color'] = 'red'
            except:
                continue
                          

    if args.out_path:
        breakpoint()
        print(f"processed {len(dags)}/{len(lines)} dags")
        pickle.dump(dags, open(args.out_path, 'wb+'))
        breakpoint()

    # G = nx.MultiDiGraph(e)
    fig = plt.Figure(figsize=(50, 50))
    ax = fig.add_subplot()
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, with_labels=True, ax=ax)
    for edge in tqdm(graph.edges(data=True)):  
        if 'weight' not in edge[2]: continue
        weight = edge[2]['weight']
        color = edge[2]['color']
        options = {'connectionstyle':'arc3,rad=0.2', 'arrowsize':100, 'width':weight, 'ax':ax, 'edge_color':[color]}
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], **options)
    fig.savefig(args.graph_vis_file)
    print(args.graph_vis_file)
