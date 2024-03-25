import argparse
import pickle
import json
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import figure, text

from .graph import *

import uuid
from pathlib import Path
from typing import Optional, Union

import rdkit.Chem as Chem
from rdkit.Chem import Draw
import torch
from tqdm import tqdm
from time import time

OPTIONS = {'connectionstyle':'arc3,rad=0.2', 'arrowsize':100}
COLORS = ' rgbcmykw'


class MolDrawer:
    """Draws molecules as images."""

    def __init__(self, path: Optional[str], subfolder: str = "assets"):

        # Init outfolder
        if not (path is not None and Path(path).exists()):
            raise NotADirectoryError(path)
        self.outfolder = Path(path) / subfolder
        self.outfolder.mkdir(exist_ok=1)

        # Placeholder
        self.lookup: dict[str, str] = None

    def _hash(self, smiles):
        """Hashing for amateurs.
        Goal: Get a short, valid, and hopefully unique filename for each molecule."""
        self.lookup = {smile: str(uuid.uuid4())[:8] for smile in smiles}
        return self

    def get_path(self) -> str:
        return self.path

    def get_molecule_filesnames(self):
        return self.lookup

    def plot(self, smiles):
        """Plot smiles as 2d molecules and save to `self.path/subfolder/*.svg`."""
        if isinstance(smiles[0], str):
            self._hash(smiles)

            for k, v in self.lookup.items():
                fname = self.outfolder / f"{v}.svg"
                mol = Chem.MolFromSmiles(k)
                # Plot
                drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 150)
                opts = drawer.drawOptions()
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                p = drawer.GetDrawingText()

                print(fname)
                with open(fname, "w") as f:
                    f.write(p)
        else:
            self._hash([Chem.MolToSmiles(m) for m in mols])
            smi_to_mol = dict(zip([Chem.MolToSmiles(m) for m in mols], mols))
            for k, v in self.lookup.items():
                fname = self.outfolder / f"{v}.svg"
                mol = smi_to_mol[k]
                # Plot
                drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 150)
                opts = drawer.drawOptions()
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                p = drawer.GetDrawingText()

                print(fname)
                with open(fname, "w") as f:
                    f.write(p)

        return self


from functools import wraps
from typing import Callable


class PrefixWriter:
    def __init__(self, file: str = None, title=''):
        self.prefix = self._default_prefix(title) if file is None else self._load(file)

    def _default_prefix(self, title):
        md = [
            f"# {title}",
            "",
        ]
        start = ["```mermaid"]
        theming = [
            "%%{init: {",
            "    'theme': 'base',",
            "    'themeVariables'a: {",
            "        'background': '#FF0000',",
            "        'primaryColor': '#ffffff',",
            "        'clusterBkg': '#ffffff',",
            "        'clusterBorder': '#ffffff',",
            "        'edgeLabelBackground':'#ffffff',",
            "        'fontSize': '20px'",
            "        }",
            "    }",
            "}%%",
        ]
        diagram_id = ["graph BT"]
        style = [
            "classDef group stroke:#00d26a,stroke-width:2px",
        ]
        return md + start + theming + diagram_id + style

    def _load(self, file):
        with open(file, "rt") as f:
            out = [l.removesuffix("\n") for l in f]
        return out

    def write(self):
        return self.prefix


class PostfixWriter:
    def write(self):
        return ["```"]


class SynTreeWriter:
    def __init__(self, prefixer=PrefixWriter(), postfixer=PostfixWriter()):
        self.prefixer = prefixer
        self.postfixer = postfixer
        self._text: list[str] = None

    def write(self, out):
        out = self.prefixer.write() + out + self.postfixer.write()
        self._text = out
        return self

    def to_file(self, file: str, text = None):
        if text is None:
            text = self._text

        with open(file, "wt") as f:
            f.writelines((l.rstrip() + "\n" for l in text))
        return None

    @property
    def text(self):
        return self.text


def subgraph(argument: str = "") -> Callable:
    """Decorator that writes a named mermaid subparagraph.

    Example output:
    ```
    subparagraph argument
        <output of function that is decorated>
    end
    ```
    """

    def _subgraph(func) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            out = f"subgraph {argument}"
            inner = func(*args, **kwargs)
            # add a tab to inner
            TAB_CHAR = " " * 4
            inner = [f"{TAB_CHAR}{line}" for line in inner]
            return [out] + inner + ["end"]

        return wrapper

    return _subgraph


def write_edge(childs, parent):
    NODE_PREFIX = "n"
    out = []
    for c in childs:
        out += [f"{NODE_PREFIX}{c} --> {NODE_PREFIX}{parent}"]
        # out += [f"{NODE_PREFIX}{parent} --> {NODE_PREFIX}{c}"]
    return out


def mermaid_write(mols, path, graph):
    drawer = MolDrawer(path)
    index_lookup = {name_group(i): mols[i-1] for i in range(1, len(mols)+1)}
    smis = []
    for node, data in graph.nodes(data=True):
        mol = index_lookup[data['name'].split(':')[0]]
        # smi = Chem.MolToSmiles(mol)
        smi = mol
        smis.append((node, smi))

    drawer.plot([smi[1] for smi in smis])
    text = []
    for node, smi in smis:
        if not isinstance(smi, str):
            smi = Chem.MolToSmiles(smi)
        name = f'"node.smiles"'
        name = f'<img src=""{drawer.outfolder.name}/{drawer.lookup[smi]}.svg"" height=75px/>'
        classdef = "group"
        info = f"n{node}[{name}]:::{classdef}"
        text += [info]

    for edge in graph.edges(data=True):
        # childs = list(graph[node])
        name_a = graph.nodes(data=True)[edge[0]]['name']
        name_b = graph.nodes(data=True)[edge[1]]['name']
        w = edge[2]['w']
        @subgraph(f'"{name_a}:{edge[0]}->{name_b}:{edge[1]}={w}"')
        def __printer():
            return write_edge(
                [edge[1]],
                edge[0],
            )

        out = __printer()
        text.extend(out)   
    return text 


def visualize_rule(args, title, dag, path):
    graph = nx.DiGraph()
    added = set()
    for (i, edge) in enumerate(dag):    
        a, b, name_a, name_b, e, w = edge
        if a not in added:
            graph.add_node(a, name=name_a)
            added.add(a)
        if b not in added:
            graph.add_node(b, name=name_b)
            added.add(b)
        graph.add_edge(a, b, e=e, w="{:.2f}".format(float(w)))

    mols = load_mols(args.motifs_folder)
    annotate_extra(mols, args.extra_label_path)  
    text = mermaid_write(mols, args.rule_vis_folder, graph)    
    SynTreeWriter(prefixer=PrefixWriter(title=title)).write(text).to_file(path)



def visualize_dag(args, dag_and_props, path):
    graph = nx.DiGraph()
    dag, *props = dag_and_props
    added = set()
    for (i, edge) in enumerate(dag):    
        a, b, name_a, name_b, e, _, w = edge
        if a not in added:
            graph.add_node(a, name=name_a)
            added.add(a)
        if b not in added:
            graph.add_node(b, name=name_b)
            added.add(b)
        graph.add_edge(a, b, e=e, w="{:.2f}".format(float(w)))

    mols = load_mols(args.motifs_folder)
    text = mermaid_write(mols, args.out_path, graph, props)    
    SynTreeWriter(prefixer=PrefixWriter(title=f"H2={props[0]}, H2/N2={props[1]}")).write(text).to_file(path)



def vis_walks_on_graph(args, graph):
    fig = plt.Figure(figsize=(50, 50))
    ax = fig.add_subplot()
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, with_labels=True, ax=ax)
    options = deepcopy(OPTIONS)
    options['ax'] = ax
    for edge in tqdm(graph.edges(data=True)):  
        if 'weight' not in edge[2]: continue
        weight = edge[2]['weight']
        color = edge[2]['color']         
        options['weight'] = weight
        options['edge_color'] = color
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], **options)
    fig.savefig(args.graph_vis_file)
    print(args.graph_vis_file)


def vis_transitions_on_graph(args, walk, states, graph):
    if len(states) != len(walk):
        breakpoint()    
    fig = plt.Figure(figsize=(100, 100))
    ax = fig.add_subplot()
    pos = nx.circular_layout(graph)
    # pos['G81'], pos['G250'] = pos['G250'], pos['G81']
    # walk_nodes = [w.val for w in walk]
    walk_nodes = [w.val for w in walk[:1]]
    node_size = [100000 if n in walk_nodes else 500 for n in graph]
    nx.draw_networkx_nodes(graph, pos, nodelist=list(graph), node_color='black', node_size=node_size, ax=ax)    
    nx.draw_networkx_labels(graph, pos, ax=ax, font_color='white', font_size=100, labels={n:n if n in walk_nodes else '' for n in graph})
    for i in range(1, len(walk)):        
        for j in range(len(states[i])):
            options = deepcopy(OPTIONS)
            options['ax'] = ax
            options['edge_color'] = COLORS[i%len(COLORS)]
            weight = states[i][j].item()
            options['width'] = min(500*weight, 100)            
            if list(graph)[j] == walk[i].val:    
                # print(weight)
                # options['edge_color'] = 'black'
                # options['arrowsize'] = 500
                # nx.draw_networkx_edges(graph, pos, edgelist=[(walk[i-1].val,list(graph)[j],0)], **options)
                nx.draw_networkx_edges(graph, pos, edgelist=[(walk[i-1].val,list(graph)[j],0)], **options)  
            elif list(graph)[j] in graph[walk[i-1].val]:
                nx.draw_networkx_edges(graph, pos, edgelist=[(walk[i-1].val,list(graph)[j],0)], **options)                            
    path = os.path.join(args.vis_folder, f"{time()}.png")
    fig.savefig(path)
    if len(walk) == 2:
        breakpoint()
    print(path)
        


def process_good_traj(traj, all_nodes):
    """
    take a numbered traj, like ['61', '90', '50[->12->37,->48]', '90']
    turn it into ['L3','S32','S20[->P14->P39,->S18]','S32']
    simple string parsing algo is enough
    """
    name_traj = []
    for x in traj:
        i = 0
        y = ""
        while i < len(x):
            if x[i].isdigit():
                j = i+1
                while j < len(x):
                    if not x[j].isdigit():
                        break
                    j += 1
                y += all_nodes[int(x[i:j])]
                i = j
            else:
                y += x[i]
                i += 1
        name_traj.append(y)
    return name_traj
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--E_file')
    parser.add_argument('--motifs_folder')
    parser.add_argument('--extra_label_path')
    parser.add_argument('--predefined_graph_file')
    parser.add_argument('--dags_file')
    parser.add_argument('--out_path')
    args = parser.parse_args()
    mols = load_mols(args.motifs_folder)
    red_grps = annotate_extra(mols, args.extra_label_path)  
    r_lookup = r_member_lookup(mols)    
    dags = load_dags(args)        
    graph = DiffusionGraph(dags, graph)     

    # dags = json.load(open(args.dags_file, 'r'))
    # os.makedirs(args.out_path, exist_ok=True)
    # for i, dag_and_props in enumerate(dags['old']):
    #     visualize_dag(args, dag_and_props, os.path.join(args.out_path, f"old_{i}.md"))
    # for i, dag_and_props in enumerate(dags['novel']):
    #     visualize_dag(args, dag_and_props, os.path.join(args.out_path, f"new_{i}.md"))
    G = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)    
    all_nodes = list(G.nodes())
    rules = ['G333', 'G393', 'G333:1']
    for i, rule in enumerate(rules):
        root, edge_conn = verify_walk(r_lookup, G, rule, loop_back='group-contrib' in os.environ['dataset'])
        visualize_dag(args, root, os.path.join(args.out_path, f"rule_{i}.md"))
        