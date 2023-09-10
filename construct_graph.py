import networkx as nx
import argparse
import matplotlib.pyplot as plt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file')
    parser.add_argument('--graph_vis_file')
    args = parser.parse_args()
    lines = open(args.data_file).readlines()    
    e = []
    nodes = []
    for l in lines:
        walk = l.split('->')
        for a, b in zip(walk[:-1], walk[1:]):
            e.append([a, b])
            e.append([b, a])
            nodes.append(a)
            nodes.append(b)
    print(nodes)
    G = nx.MultiDiGraph(e)
    fig = plt.Figure(figsize=(20, 20))
    nx.draw(G, with_labels=True, ax=fig.add_subplot())    
    fig.savefig(args.graph_vis_file)