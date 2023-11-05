import argparse
import pickle
import json
import networkx as nx
import os
from graph import *

import uuid
from pathlib import Path
from typing import Optional, Union

import rdkit.Chem as Chem
from rdkit.Chem import Draw
import torch


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
        out += [f"{NODE_PREFIX}{c} <--> {NODE_PREFIX}{parent}"]
    return out


def mermaid_write(mols, path, graph, props):
    drawer = MolDrawer(path)
    index_lookup = {name_group(i): mols[i-1] for i in range(1, 98)}
    smis = []
    for node, data in graph.nodes(data=True):
        mol = index_lookup[data['name']]
        smi = Chem.MolToSmiles(mol)
        smis.append((node, smi))

    drawer.plot([smi[1] for smi in smis])
    text = []
    for node, smi in smis:
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
        @subgraph(f'"{name_b}:{edge[1]}->{name_a}:{edge[0]}={w}"')
        def __printer():
            return write_edge(
                [edge[1]],
                edge[0],
            )

        out = __printer()
        text.extend(out)   
    return text 



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--E_file')
    parser.add_argument('--motifs_folder')
    parser.add_argument('--predefined_graph_file')
    parser.add_argument('--walks_file')
    parser.add_argument('--dags_file')
    parser.add_argument('--out_path')
    args = parser.parse_args()

    dags = json.load(open(args.dags_file, 'r'))
    os.makedirs(args.out_path, exist_ok=True)
    for i, dag_and_props in enumerate(dags['old']):
        visualize_dag(args, dag_and_props, os.path.join(args.out_path, f"old_{i}.md"))
    for i, dag_and_props in enumerate(dags['novel']):
        visualize_dag(args, dag_and_props, os.path.join(args.out_path, f"new_{i}.md"))
        