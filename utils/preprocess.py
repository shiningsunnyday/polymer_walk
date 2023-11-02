import os
from rdkit import Chem
from rdkit.Chem import Draw
from collections import defaultdict
import re
import networkx as nx
from data import *

def name_group(m):
    # return f"G{m}" # not group contrib
    prefix = lambda x: "P" if x <= 41 else ("S" if x <= 73 else "L")
    return prefix(m)+f"{m if m<=41 else m-41 if m<=73 else m-73}"



def load_mols(folder):
    mols = []
    files = os.listdir(folder)
    files = list(filter(lambda x: x.endswith('.mol'), files))
    files = sorted(files, key=lambda x:int(x.rstrip('.mol')))
    for f in files:
        # if f == '15.mol':
        #     mol = Chem.MolFromSmiles('CC1=CC(C)=CC=C1C')
        #     breakpoint()
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))            
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/all_groups/with_label/15.mol')
        if f.endswith('.mol'):
            mol = Chem.rdmolfiles.MolFromMolFile(os.path.join(folder, f))
            if not mol:
                mol = Chem.rdmolfiles.MolFromMolFile(os.path.join(folder, f), sanitize=False)
            assert mol
            mols.append(mol)
            for i, a in enumerate(mol.GetAtoms()): 
                a.SetProp("molAtomMapNumber", str(i+1))
    return mols


def load_walks(args):
    if os.path.isdir(args.data_file):
        for f in os.listdir(args.data_file):
            if f.endswith('.edgelist'):
                graph = nx.read_edgelist(os.path.join(args.data_file, f))
                pass
    else:
        lines = open(args.data_file).readlines()
        walks = set()
        for i, l in enumerate(lines):        
            walk = l.rstrip('\n').split(' ')[0]
            walks.add(walk)
        print(walks)    
    return walks


def r_member_lookup(mols):
    dic = defaultdict(lambda: defaultdict(list))
    for i, mol in enumerate(mols):
        for j, a in enumerate(mol.GetAtoms()):
            r = a.GetProp('r_grp')
            if not r: continue
            for r_ in r.split('-'):
                dic[name_group(i+1)][r_].append(j)
            
    return dic



def annotate_extra(mols, path):
    all_labels = []
    labels = []
    volumes = []
    red_grps = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if line.startswith('Volume:'):
                volumes = line[len('Volume:'):].rstrip('\n').split(',')
                continue
            if not line:
                break
            if line == '\n':
                if labels:
                    all_labels.append(labels) # labels can be [] if mols has no atoms
                labels = []
            else:
                red_grp = ''
                label = line.split(':')[0]
                if ':' in line:
                    red_grp = line.split(':')[1]
                else:
                    red_grp = ''

                labels.append((int(label), red_grp.rstrip('\n')))

    print(len(mols), len(all_labels))
    if volumes: assert len(volumes) == len(mols)

    for i in range(len(all_labels)):
        m = mols[i]
        if volumes: m.GetConformer().SetProp('V', volumes[i])
        labels = [l[0] for l in all_labels[i]]
        red_grps = [l[1] for l in all_labels[i]]
        for i in range(m.GetNumAtoms()):
            m.GetAtomWithIdx(i).SetBoolProp('r', False)
            m.GetAtomWithIdx(i).SetProp('r_grp', '')
        for i in range(m.GetNumBonds()):
            m.GetBondWithIdx(i).SetBoolProp('r', False)
            m.GetBondWithIdx(i).SetBoolProp('picked', False)
        for l, red_grp in zip(labels, red_grps):
            if l <= m.GetNumAtoms() and red_grp:
                a = m.GetAtomWithIdx(l-1)
                a.SetBoolProp('r', True)
                a.SetProp('r_grp', red_grp)
                a.SetProp('atomLabel', f"R_{a.GetSymbol()}{l}")
                for b in a.GetBonds():
                    b.SetBoolProp('r', True)
    return red_grps
