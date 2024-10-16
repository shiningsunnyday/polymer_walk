import sys
import argparse 
from rdkit import Chem
from rdkit.Chem import rdchem
from multiprocessing import Pool
from tqdm import tqdm
from itertools import permutations
from functools import reduce
import os
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pyjokes
from rdkit.Chem.rdmolops import FastFindRings

def __extract_subgraph(mol, selected_atoms):
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetIntProp('org_idx', atom.GetIdx())

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots

def detect_ring(mol, atoms):
    atom_idxs = [a-1 for a in atoms]
    # are either all part of ring or not
    in_rings = []
    for ring in mol.GetRingInfo().AtomRings():
        part_of = [a in ring for a in atom_idxs]
        if np.all(part_of) == len(atom_idxs):
            in_rings.append(ring)
        elif np.all(part_of) > 0:
            raise AssertionError(f"some (but not all) atoms part of ring {ring}")
    if len(in_rings) > 1:
        raise AssertionError(f"atoms entirely contained in >1 rings")
    elif len(in_rings) == 1:
        return in_rings
    else:
        return None
    
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)
    
def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol
    

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        #if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
        #    bt = Chem.rdchem.BondType.SINGLE
    return new_mol
    

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) 
    #if tmp_mol is not None: new_mol = tmp_mol
    return new_mol

            
def find_fragments(mol):
    new_mol = Chem.RWMol(mol)
    removed_bonds = []
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    for bond in mol.GetBonds():
        if bond.IsInRing(): continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if a1.IsInRing() and a2.IsInRing():
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
            removed_bonds.append((a1.GetIdx(), a2.GetIdx()))

        elif a1.IsInRing() and a2.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(a1))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
            new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
            removed_bonds.append((a1.GetIdx(), a2.GetIdx()))

        elif a2.IsInRing() and a1.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(a2))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a2.GetIdx())
            new_mol.AddBond(new_idx, a1.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
            removed_bonds.append((a1.GetIdx(), a2.GetIdx()))
    
    new_mol = new_mol.GetMol()
    new_smiles = Chem.MolToSmiles(new_mol)

    hopts = []
    for fragment in new_smiles.split('.'):
        fmol = Chem.MolFromSmiles(fragment)
        indices = set([atom.GetAtomMapNum() for atom in fmol.GetAtoms()])
        fmol = get_clique_mol(mol, indices)
        fmol = sanitize(fmol, kekulize=False)
        fsmiles = Chem.MolToSmiles(fmol)
        hopts.append((fsmiles, indices))
    
    return hopts, removed_bonds


def break_bonds(mol, bonds, red_grps):
    assert len(bonds) == len(red_grps), "number of bonds != number of red groups"
    for bs, (red_grp1, red_grp2) in zip(bonds, red_grps):
        for b in bs:
            assert len(b) == 2, "bond is not exactly 2 atoms"
            b1, b2 = b
            assert (red_grp1==[]) == (red_grp2==[]), "specify none or both of red groups"
            if red_grp1:
                assert b1 in red_grp1, "first atom in bond not in first red group"
            if red_grp2:
                assert b2 in red_grp2, "second atom in bond not in second red group"
    mol = Chem.RWMol(mol)        
    for bs in bonds:
        for b in bs:
            b1, b2 = b
            assert mol.GetBondBetweenAtoms(b1-1, b2-1) is not None, f"bond {b} is None!"
            mol.RemoveBond(b1-1, b2-1)    
    frags = Chem.GetMolFrags(mol) 
    frags = [[f+1 for f in frag] for frag in frags]
    frag_str = [','.join(map(str, frag)) for frag in frags]    
    occur = [False for _ in frag_str]
    for bs, (red_grp1, red_grp2) in zip(bonds, red_grps):
        indices = [-1, -1]
        rings = [None, None]
        for i in range(len(frags)):
            bs_1 = [b1 for b1, _ in bs]
            bs_2 = [b2 for _, b2 in bs]

            if np.all([b1 in frags[i] for b1 in bs_1]):
                indices[0] = i
                if red_grp1 == []:
                    if len(frags[i]) == 1:
                        red_grp1 = detect_ring(mol, bs_1)
                        if red_grp1 is None:
                            red_grp1 = bs_1
                    else:
                        red_grp1 = bs_1
            elif np.all([b2 in frags[i] for b2 in bs_2]):
                indices[1] = i    
                if red_grp2 == []:
                    if len(frags[i]) == 1:
                        red_grp2 = detect_ring(mol, bs_1)
                        if red_grp2 is None:
                            red_grp2 = bs_2
                    else:
                        red_grp2 = bs_2

        assert min(indices) != -1, "segmentation is not valid!"
        for j in range(2):
            delim = ';' if occur[indices[j]] else ':'
            frag_str[indices[j]] += delim + ','.join(map(str, red_grp1 if j else red_grp2))
            occur[indices[j]] = True
    return frag_str


def mol_to_graph(mol, inds, r=False):
    graph = nx.Graph()
    for i, ind in enumerate(inds):
        a = mol.GetAtomWithIdx(ind)
        kwargs = {}
        if r: kwargs['r'] = a.GetBoolProp('r')
        graph.add_node(str(i), symbol=a.GetSymbol(), **kwargs)
    for i, u in enumerate(inds):
        for j, v in enumerate(inds):
            bond = mol.GetBondBetweenAtoms(u, v)
            if bond:
                graph.add_edge(str(i), str(j), bond_type=bond.GetBondType())
    return graph
        

def check_order(orig_mol, mol, cls, r=False):
    """
    Check whether subgraph induced by cls ~ mol
    r further checks if red atoms are the same
    """    
    
    graph_1 = mol_to_graph(orig_mol, cls, r=r)
    graph_2 = mol_to_graph(mol, list(range(mol.GetNumAtoms())), r=r)
    def node_match(a, b):
        return (a['symbol'] == b['symbol']) and ((not r) or (a['r'] == b['r']))    
    def edge_match(ab, cd):
        return ab['bond_type'] == cd['bond_type']
    if len(graph_1) != len(graph_2):
        return False
    if len(graph_1) == 1:
        res = list(dict(graph_1.nodes(data=True)).values())[0] == list(dict(graph_2.nodes(data=True)).values())[0]
    else:
        res = nx.is_isomorphic(graph_1, graph_2, node_match, edge_match)
    
    return res

    # below is much slower, but just to check
    ans = False
    for cluster in permutations(cls):
        bad = False
        for i in range(len(cluster)):
            if orig_mol.GetAtomWithIdx(cluster[i]).GetSymbol() != mol.GetAtomWithIdx(i).GetSymbol():
                bad = True
                continue
            if r and orig_mol.GetAtomWithIdx(cluster[i]).GetBoolProp('r') != mol.GetAtomWithIdx(i).GetBoolProp('r'):
                bad = True
                continue            
            for j in range(len(cluster)):
                bond = orig_mol.GetBondBetweenAtoms(cluster[i], cluster[j])
                bond_ = mol.GetBondBetweenAtoms(i, j)
                if (bond == None) ^ (bond_ == None): 
                    bad = True
                    break
                if bond:
                    if bond.GetBondType() != bond_.GetBondType():
                        bad = True
                        break  
            if bad:
                break
        if not bad:
            ans = True
            break
    if res != ans:
        breakpoint()
    return ans


def induce_mol(old_mol, cluster):
    # subgraph = __extract_subgraph(old_mol, cluster)[0]
    mol = Chem.MolFromSmiles('')
    ed_mol = Chem.EditableMol(mol)
    for i, c in enumerate(cluster):
        ed_mol.AddAtom(rdchem.Atom(old_mol.GetAtomWithIdx(c).GetSymbol()))
    mol = ed_mol.GetMol()
    for i, c1 in enumerate(cluster):
        for j, c2 in enumerate(cluster):
            if i > j: continue
            b = old_mol.GetBondBetweenAtoms(c1, c2)
            if b:
                ed_mol.AddBond(i, j, b.GetBondType())
    
    mol = ed_mol.GetMol()
    # """
    # Assert the subgraph and mol are the same
    # """    
    # for i in range(mol.GetNumAtoms()):
    #     assert mol.GetAtomWithIdx(i).GetSymbol() == subgraph.GetAtomWithIdx(i).GetSymbol()
    #     for j in range(mol.GetNumAtoms()):
    #         b1 = mol.GetBondBetweenAtoms(i,j)
    #         b2 = subgraph.GetBondBetweenAtoms(i,j)
    #         assert (b1 is None) == (b2 is None)
    #         if b1 is not None:
    #             assert b1.GetBondTypeAsDouble() == b2.GetBondTypeAsDouble()
    if not check_order(old_mol, mol, cluster):
        breakpoint()
    return mol


def annotate_extra_mol(m, labels, red_grps):
    """
    labels: list of atom id's with red grps
    red_grps: list of hyphen-separated red group id's
    """
    for i, a in enumerate(m.GetAtoms()): 
        a.SetProp("molAtomMapNumber", str(i+1))    
    for i in range(m.GetNumAtoms()):
        m.GetAtomWithIdx(i).SetBoolProp('r', False)
        m.GetAtomWithIdx(i).SetProp('r_grp', '')
    for i in range(m.GetNumBonds()):
        m.GetBondWithIdx(i).SetBoolProp('r', False)
        m.GetBondWithIdx(i).SetBoolProp('picked', False)
    for l, red_grp in zip(labels, red_grps):
        if l <= m.GetNumAtoms():
            a = m.GetAtomWithIdx(l-1)
            a.SetBoolProp('r', True)
            a.SetProp('r_grp', red_grp)
            a.SetProp('atomLabel', f"R{red_grp}_{a.GetSymbol()}{l}")
            for b in a.GetBonds():
                b.SetBoolProp('r', True)    


def draw_hmol(path, G):
    fig = plt.Figure(figsize=(10, 10))
    ax = fig.add_subplot()
    options = {"node_size": 500}
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes, ax=ax, **options)    
    nx.draw_networkx_edges(G, pos, ax=ax)    
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    fig.savefig(path)


# def dfs(i, G, walk):
#     walk.append(i)
#     for j in G[i]:
#         if j in walk:
#             continue
#         dfs(j, G, walk)

# dfs(0, G, walk)   

def dfs(i, hmol, edges, vis):
    vis[i] = True
    for j in hmol.mol_tree[i]:
        if vis[j]: 
            continue
        breakpoint()
        if len(hmol.clusters[j]) == 1:
            for k in hmol.mol_tree[j]:    
                if vis[k]: continue
                edges.append((i, k))
            vis[j] = True
            for k in hmol.mol_tree[j]:
                if vis[k]: continue
                dfs(k, hmol, edges, vis)
                
        else:
            edges.append((i, j))
            dfs(j, hmol, edges, vis)


def dfs_explore(i, hmol, vis, explored):
    vis[i] = True
    hmol.clusters[i] = tuple(a for a in hmol.clusters[i] if a not in explored)
    explored |= set(hmol.clusters[i])
    for j in hmol.mol_tree[i]:
        if vis[j]: 
            continue
    
        dfs_explore(j, hmol, vis, explored)

def extract_groups(hmol, edges, node_label, vocab_mols, l):
    for edge in edges:
        a = edge[0]
        b = edge[1]
        
        mol_a = induce_mol(hmol.mol, hmol.clusters[a])
        cluster_a = hmol.clusters[a]       
        # cluster_a = check_order(hmol, vocab_mols[smi_a], hmol.clusters[a])
        mol_b = induce_mol(hmol.mol, hmol.clusters[b])
        cluster_b = hmol.clusters[b]                
        ab = list(set(cluster_a) | set(cluster_b)) # red
        if len(ab) == max(len(cluster_a), len(cluster_b)):
            assert min(len(cluster_a), len(cluster_b)) == 1
            breakpoint()
            continue
        
        ab_mol = induce_mol(hmol.mol, ab)
        cluster = set(cluster_b)-set(cluster_a) # a is red              
        canon_smi = Chem.CanonSmiles(Chem.MolToSmiles(ab_mol))                            
        if canon_smi in vocab_mols:
            orig_mol = vocab_mols[canon_smi][2].mol
            prev_cluster_a = vocab_mols[canon_smi][3]
            cur_mol_a = induce_mol(hmol.mol, cluster_a)
            breakpoint()
            if check_order(orig_mol, cur_mol_a, prev_cluster_a):
                if a in node_label:
                    node_label[a] += f"-{vocab_mols[canon_smi][0]}"
                else:
                    node_label[a] = str(vocab_mols[canon_smi][0])
            else:
                breakpoint()
        else:                              
            if a in node_label: 
                node_label[a] += f"-{str(l)}"
            else:
                node_label[a] = str(l)

        # set node with group
        for j, at in enumerate(ab):
            ab_mol.GetAtomWithIdx(j).SetBoolProp('r', at in cluster)
        
        # indices of red atoms
        labels = [j+1 for j in range(ab_mol.GetNumAtoms()) if ab[j] in cluster]
        red_grps = ['1' for _ in labels]
        annotate_extra_mol(ab_mol, labels, red_grps)   
        if canon_smi not in vocab_mols:
            vocab_mols[canon_smi] = (l, ab_mol, hmol, cluster_a, cluster_b)
            l += 1
    return l


def extract_clusters(hmol, edges, vocab_mols, l):
    node_label = {}
    for node in hmol.mol_tree.nodes():
        # can we be dom? check for any dom neighbors
        # dom_nei = False
        # for nei in hmol.mol_tree[node]:
        #     dom_nei |= dom.get(nei, False)        
        # dom[node] = not dom_nei
        dic = {}
        cluster = set(hmol.clusters[node])
        nei_cluster = set()
        for nei in hmol.mol_tree[node]:
            cluster |= set(hmol.clusters[nei])
            nei_cluster |= set(hmol.clusters[nei])
        cluster = list(cluster)
        mol = induce_mol(hmol.mol, cluster)
        labels = [] # indices of atoms which are red
        red_grps = [] # for each atom in labels, which group(s) they're in
        for a in mol.GetAtoms():   
            red = 0
            if cluster[a.GetIdx()] not in hmol.clusters[node]:
                for nei in hmol.mol_tree[node]:
                    if cluster[a.GetIdx()] in hmol.clusters[nei]:
                        if red:
                            breakpoint()
                        red = nei+1
            if red:
                a.SetBoolProp('r', True)
                labels.append(a.GetIdx()+1)
                if red not in dic:                    
                    dic[red] = str(len(dic)+1)
                red_grp = dic[red]
                red_grps.append(red_grp)
            else:
                a.SetBoolProp('r', False)
       
        annotate_extra_mol(mol, labels, red_grps)
        try:
            canon_smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        except:
            canon_smi = Chem.MolToSmiles(mol)

        vocab_i = -1
        for i, (k, vocab_mol, _) in enumerate(vocab_mols):
            # check isomorphism
            if mol.GetNumAtoms() != vocab_mol.GetNumAtoms():
                continue
                
            if check_order(mol, vocab_mol, list(range(mol.GetNumAtoms())), r=True):
                vocab_i = i
                break
                
        if vocab_i > -1:
            node_label[node] = str(vocab_mols[vocab_i][-1])
        else:                 
            vocab_mols.append((canon_smi, mol, l))
            node_label[node] = str(l)
            l += 1

    counts = {} 
    for k,v in node_label.items():
        counts[v] = counts.get(v, 0)+1
        node_label[k] = f"G{v}"
        if counts[v]>1:
            node_label[k] += f":{counts[v]-1}"
    hmol.mol_tree = nx.relabel_nodes(hmol.mol_tree, node_label)
    return l



def draw_grid(pics, titles, fpath=""):
    grid_dim = 1+int(np.sqrt(len(pics)-1))
    f, axes = plt.subplots(grid_dim, (len(pics)-1)//grid_dim+1, figsize=(100,100))
    grid_dim = (len(pics)-1)//grid_dim+1
    for i in range(len(pics)):
        axes[i//grid_dim][i%grid_dim].imshow(Image.open(pics[i]))
        axes[i//grid_dim][i%grid_dim].set_title(titles[i])
        axes[i//grid_dim][i%grid_dim].set_xticks([])
        axes[i//grid_dim][i%grid_dim].set_yticks([])
    if fpath:
        f.savefig(fpath, dpi=200.)    

            
            

def preprocess(smi):
    mol = Chem.MolFromSmiles(smi)
    for i in range(mol.GetNumAtoms()-1,-1,-1):
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() == '*':
            ed_mol = Chem.EditableMol(mol)
            ed_mol.RemoveAtom(i)
            mol = ed_mol.GetMol()
    return Chem.MolToSmiles(mol)



def main(args):
    data = []
    lines = open(args.data_file).readlines()
    # data = [preprocess(l.split(',')[0]) for l in lines]
    data = [l.split(',')[0] for l in lines]

    vocab = seg_groups(args)
 
    os.makedirs(os.path.join(args.group_dir, f'all_groups/with_label'), exist_ok=True)
    f = open(os.path.join(args.group_dir, "group_smiles.txt"), 'w+')        
    extra = ""
    for smi, mol, i in vocab:
        # f.write(f"{smi}\n")
        for a in mol.GetAtoms():
            r_grp = a.GetProp('r_grp')
            if r_grp:
                extra += f"{a.GetIdx()+1}:{r_grp}\n"
            else: 
                extra += f"{a.GetIdx()+1}\n"
        extra += "\n"
        try:
            Chem.rdmolfiles.MolToMolFile(mol, os.path.join(args.group_dir, f"all_groups/{i}.mol"))
        except:
            Chem.rdmolfiles.MolToMolFile(mol, os.path.join(args.group_dir, f"all_groups/{i}.mol"), kekulize=False)
        Draw.MolToFile(mol, os.path.join(args.group_dir, f'all_groups/with_label/{i}.png'), size=(2000, 2000))
        f.write(f"{smi}\n")
    f.close()
    with open(os.path.join(args.group_dir, "all_groups/all_extra.txt"), 'w+') as f:
        f.write(extra) 


def infer_nei_ring(mol, cluster_neis, extra)  :
    nei_ring = None
    for ring in mol.GetRingInfo().AtomRings():                                
        if np.all([c in ring for c in cluster_neis]):
            contained = True
            for r in ring:
                if r+1 not in extra:
                    contained = False
            if contained:
                if nei_ring:
                    raise
                nei_ring = [r+1 for r in ring]
    return nei_ring


def seg_mol(mol, mol_segs, vocab_mols, l, index=-1, annotated=False):
    node_label = {}
    edges = []
    global_to_local_idxes = {}    
    clusters = []
    new_mol_segs = [[None, None] for _ in mol_segs] # cluster, red_bond_info
    for i, g1 in enumerate(mol_segs):            
        extras = []
        edges_i = []
        if ':' in g1:
            if len(g1.split(':')) == 2:
                cluster, red_bond_info = g1.split(':')

            else:
                print(f"{g1} bad syntax")            
                return l, None
        else:
            cluster = g1
            if len(mol_segs) != 1:
                print(f"if {g1} has no red atoms, there should be only one segment")
                return l, None               
            else:
                red_bond_info = ''
        new_mol_segs[i][0] = cluster
        
        cluster_split = cluster.split(',')
        # syntax-check cluster and red_bond_info
        for c in cluster_split:
            if not c.isdigit():
                print(f"{cluster} bad syntax")            
                return l, None
        if len(set(cluster_split)) != len(cluster_split):
            c_counts = Counter(cluster_split)
            dups = [k for k, v in c_counts.items() if v > 1]
            print(f"remove duplicate {dups} from {cluster_split}")            
            return l, None
        if red_bond_info:
            for red_bond_group in red_bond_info.split(';'):
                for r in red_bond_group.split(','):
                    if not r.isdigit():
                        print(f"{red_bond_info} bad syntax")
                        return l, None
                
                       
        if not cluster:
            print(f"{g1} bad syntax")            
            return l, None            
       
        if red_bond_info:
            cluster = set(map(int, cluster.split(',')))
            given_extras = red_bond_info.split(';')
            new_mol_segs[i][1] = given_extras
            for k, e in enumerate(given_extras):
                extra = list(map(int, e.split(',')))
                if not annotated:
                    if len(extra) > 6:
                        cluster_neis = [nei.GetIdx()+1 for c in cluster for nei in mol.GetAtomWithIdx(c-1).GetNeighbors()]
                        cluster_neis = [c-1 for c in cluster_neis if c in extra]
                        if len(cluster_neis) == 1:
                            if mol.GetAtomWithIdx(cluster_neis[0]).IsInRing():
                                try:
                                    nei_ring = infer_nei_ring(mol, cluster_neis, extra)
                                except:
                                    print("ambiguity in red group ring")
                                    return l, None
                                if len(nei_ring) not in [5,6]:
                                    extra = [c+1 for c in cluster_neis]
                                elif nei_ring is None:
                                    breakpoint()
                                else:
                                    extra = [e for e in extra if e in nei_ring]                            
                            else:
                                extra = [c+1 for c in cluster_neis]
                        elif len(cluster_neis) == 2:
                            if np.all([mol.GetAtomWithIdx(c).IsInRing() for c in cluster_neis]):
                                nei_ring = infer_nei_ring(mol, cluster_neis, extra)
                                if nei_ring is None:         
                                    pass
                                    # extra = [c+1 for c in cluster_neis]
                                else:
                                    extra = [e for e in extra if e in nei_ring]
                            else:
                                breakpoint()
                        else:
                            breakpoint()
                        # print(f"{extra} has >6 atoms, make sure red group is not over-specified")
                        # return l, None
                    elif len(extra) != 1:
                        cluster_neis = [nei.GetIdx()+1 for c in cluster for nei in mol.GetAtomWithIdx(c-1).GetNeighbors()]
                        cluster_neis = [c-1 for c in cluster_neis if c in extra]
                        if len(cluster_neis) == 1:
                            if mol.GetAtomWithIdx(cluster_neis[0]).IsInRing():
                                nei_ring = infer_nei_ring(mol, cluster_neis, extra)
                                if nei_ring is None:
                                    extra = [c+1 for c in cluster_neis]
                                else:
                                    extra = [e for e in extra if e in nei_ring]                            
                            else:
                                extra = [c+1 for c in cluster_neis]
                e_atoms = set(extra)
                new_mol_segs[i][1][k] = ','.join(map(str,extra))
                extras.append(extra)                
                for j, g2 in enumerate(mol_segs):
                    try:
                        extra_cluster, _ = g2.split(':')
                    except:
                        print(f"{g2} bad syntax")                    
                        return l, None
                    if extra_cluster == '':
                        print(f"{g2} bad syntax")                    
                        return l, None                        
                    try:
                        extra_cluster = set(map(int, extra_cluster.split(',')))    
                    except:
                        print(f"{extra_cluster} bad syntax")
                        return l, None
                    if extra_cluster & e_atoms:
                        if extra_cluster & e_atoms != e_atoms:
                            print(f"{extra} intersects {g2.split(':')[0]} at {extra_cluster & e_atoms} but is not entirely contained within {g2.split(':')[0]}")                        
                            return l, None                    
                        edges_i.append([i, j, [extras[-1]]])
                        break
                    if j == len(mol_segs)-1:
                        print(f"seg {i} extra {extra} is not among black atom sets")
                        return l, None
            for exist_cluster in clusters:
                if cluster & exist_cluster:     
                    print(f"{cluster} should not intersect existing {exist_cluster} at {cluster & exist_cluster}")
                    return l, None
            clusters.append(cluster)                
            cluster, red_bond_info = g1.split(':')
            cluster = list(map(int, cluster.split(',')))
        else:
            cluster = list(map(int, cluster.split(',')))        
            clusters.append(set(cluster))
        extra_atoms = [e for extra in extras for e in extra]
        extra_set = set(extra_atoms)            
        if len(extra_set) != len(extra_atoms):            
            for c, v in Counter(extra_atoms).items():
                if v>1:
                    print(f"{c} should appear at most once in {red_bond_info}")
            return l, None
        intersect_atoms = set(cluster) & set(extra_atoms)
        if intersect_atoms:            
            print(f"red {extra_atoms} should not intersect black atom set {cluster} at {intersect_atoms}")
            return l, None
        
        for idx in cluster+extra_atoms:
            if idx > mol.GetNumAtoms():
                print(f"{idx} should not exceed mol's number of atoms ({mol.GetNumAtoms()})")
                return l, None
        group_idxs = [idx-1 for idx in cluster + extra_atoms]
        group = induce_mol(mol, group_idxs)        
        for k in range(len(group_idxs)):
            ch_before = mol.GetAtomWithIdx(group_idxs[k]).GetFormalCharge()
            ch_after = group.GetAtomWithIdx(k).GetFormalCharge()
            if ch_before != ch_after:
                if index == 120:
                    group_smi = Chem.MolToSmiles(group)
                    smi = Chem.MolToSmiles(mol)
                    print(f"correcting formal charge in group {group_smi} of mol {smi}")
                    group.GetAtomWithIdx(k).SetFormalCharge(ch_before)
        frag_smi = Chem.MolToSmiles(group)
        global_to_local_idxes[i] = dict(zip(cluster + extra_atoms, range(group.GetNumAtoms())))
        local_idx_map = dict(zip(cluster+extra_atoms, range(mol.GetNumAtoms())))
        for _, _, edge_data in edges_i:
            mapped_idxes = []
            for global_idx in edge_data[0]:
                mapped_idxes.append(local_idx_map[global_idx])
            edge_data.append(mapped_idxes)  

        for k in range(len(cluster)):
            group.GetAtomWithIdx(k).SetBoolProp('r', False)        
        for k in range(len(cluster), len(cluster)+len(extra_atoms)):
            group.GetAtomWithIdx(k).SetBoolProp('r', True)            
        labels = list(range(len(cluster)+1, len(cluster)+len(extra_atoms)+1))
        red_grps = []
        for ind, extra in enumerate(extras):
            for e in extra:
                red_grps.append(str(ind+1))            
        annotate_extra_mol(group, labels, red_grps)
        vocab_i = -1
        # if index == 78:
        #     breakpoint()        
        for k, (_, vocab_mol, _) in enumerate(vocab_mols):
            # check isomorphism
            if group.GetNumAtoms() != vocab_mol.GetNumAtoms():
                continue                
            if check_order(group, vocab_mol, list(range(group.GetNumAtoms())), r=True):
                vocab_i = k
                break
        if vocab_i > -1:
            node_label[i] = str(vocab_mols[vocab_i][-1])
        else:              
            vocab_mols.append((frag_smi, group, l))
            node_label[i] = str(l)
            l += 1      
        edges += edges_i            
    new_seg_str = ""
    for k, (new_black, new_red) in enumerate(new_mol_segs):
        new_seg_str += f"{index} "
        new_seg_str += new_black
        if new_red is not None:
            new_seg_str += ":" + ';'.join(new_red)
        if k != len(new_mol_segs)-1:
            new_seg_str += "\n"
    
    # print(new_seg_str)

    counts = {} 
    for k,v in node_label.items():
        counts[v] = counts.get(v, 0)+1
        node_label[k] = f"G{v}"
        if counts[v]>1:
            node_label[k] += f":{counts[v]-1}"

    graph = nx.DiGraph()
    for i in range(len(mol_segs)):
        graph.add_node(node_label[i])    
    
    """
    We have [(i, j, r_grp_1_global, r_grp_1_local)] and vice-versa
    Create i->j with r_grp_1, r_grp_2, b_2, b_1
    """    
    edge_data_lookup = {}
    for edge in edges:
        edge_data_lookup[tuple(edge[:2])] = edge[2]
    for src, dest, (global_idx_1, r_grp_1) in edges:
        """
        Process the inter-motif edge from src->dest
        global_idx_1: idxes in original mol of group src's extra atoms
        r_grp_1: idxes in group src of its extra atoms
        """
        try:
            if graph.has_edge(node_label[dest], node_label[src]): 
                continue
        except:
            breakpoint()
        if [global_to_local_idxes[src][idx] for idx in global_idx_1] != r_grp_1:
            breakpoint()
        b2 = [global_to_local_idxes[dest][idx] for idx in global_idx_1]
        if (dest, src) not in edge_data_lookup:
            print(f"segment {mol_segs[dest]} needs to connect back to {mol_segs[src]}")
            return l, None
        global_idx_2, r_grp_2 = edge_data_lookup[(dest, src)]
        b1 = [global_to_local_idxes[src][idx] for idx in global_idx_2]
        # if index == 78 and node_label[src] == 'G101' and node_label[dest] == 'G69':
        #     breakpoint()
        graph.add_edge(node_label[src], node_label[dest], r_grp_1=r_grp_1, b2=b2, r_grp_2=r_grp_2, b1=b1)
        graph.add_edge(node_label[dest], node_label[src], r_grp_1=r_grp_2, b2=b1, r_grp_2=r_grp_1, b1=b2)                            

    if not nx.is_strongly_connected(graph):        
        print(f"graph {graph.edges()} isn't connected")
        for i in range(len(graph)):
            for j in range(len(graph)):
                if i == j:
                    continue
                if graph.has_edge(node_label[i], node_label[j]):
                    continue
                group_i = list(map(int, mol_segs[i].split(':')[0].split(',')))
                group_j = list(map(int, mol_segs[j].split(':')[0].split(',')))
                adjacent = False
                for a in group_i:
                    for b in group_j:                        
                        bond = mol.GetBondBetweenAtoms(a-1, b-1)
                        if bond is not None:
                            adjacent = True
                if adjacent:
                    print(f"{mol_segs[i]} should connect with {mol_segs[j]}")
                    seg_j = mol_segs[j].split(':')[0]
                    mol_segs[i] += f';{seg_j}'                            
        print('\n')
        print('\n'.join([f"{index} "+seg for seg in mol_segs]))
        print('\n')
        return l, None
    not_black = set(range(1, mol.GetNumAtoms()+1)) - reduce(lambda x,y: x|y, clusters)
    if not_black:        
        print(f"black atoms don't add up to {list(range(1, mol.GetNumAtoms()+1))}, where is {not_black}?")
        return l, None        
   
    # print(" ".join([mol_seg.split(":")[0] for mol_seg in mol_segs]))
    # return l, True
    return l, graph



def process_annotation(mol, info):        
    """
    a few remaining cases
    """ 
    bonds = []
    red_grps = []
    to_list = lambda x: list(map(int, x.split(','))) if x != ';' else []
    for *bs, r_grp1, r_grp2 in info:
        bonds.append([to_list(b) for b in bs])
        if len(bonds[-1]) > 2:
            breakpoint()
        if bonds[-1] == []:
            breakpoint()
        red_grps.append((to_list(r_grp1), to_list(r_grp2)))    
    try:
        return break_bonds(mol, bonds, red_grps)
    except Exception as e:
        raise e


def bond_between_fragments(mol, frag1, frag2):
    bonds = []
    for i in range(len(frag1)):
        for j in range(i+1,len(frag2)):
            if mol.GetBondBetweenAtoms(frag1[i],frag2[j]) is not None:
                bonds.append((frag1[i],frag2[j]))
    return bonds

    

def fragment_mol(mol):
    fragments, removed_bonds = find_fragments(mol)    
    mol = Chem.RWMol(mol)
    for a, b in removed_bonds:
        mol.RemoveBond(a, b)
    frags = Chem.GetMolFrags(mol)
    frags = [[f+1 for f in frag] for frag in frags]
    frag_lookup = {}
    for index, frag in enumerate(frags):
        for f in frag:
            frag_lookup[f] = index
    removed_bonds = [(a+1,b+1) for (a,b) in removed_bonds]
    removed_bonds_by_fragment = []
    while removed_bonds:
        last_bond = removed_bonds.pop(-1)
        a1, b1 = last_bond
        removed_bonds_by_fragment.append([f'{a1},{b1}'])
        for bond in removed_bonds:
            a, b = bond
            if frag_lookup[a1] == frag_lookup[a] and frag_lookup[b1] == frag_lookup[b]:
                removed_bonds_by_fragment[-1].append(f'{a},{b}')
    # seg = []    
    # for bond in removed_bonds_by_fragment:
    #     a, b = bond
    #     seg.append(','.join(map(str,bond)))
    #     if len(frags[frag_lookup[b]]) == 1:
    #         breakpoint()
    #     else:
    #         frags[frag_lookup[b]]
    #     if len(frag_lookup[a]) == 1:
    #         breakpoint()
    #     else:
    #         breakpoint()          
    for i in range(len(removed_bonds_by_fragment)):
        removed_bonds_by_fragment[i].append(';')
        removed_bonds_by_fragment[i].append(';')
    return removed_bonds_by_fragment


def seg_groups(args):
    vocab_mols = []
    l = 1    
    hmol_fig_dir = os.path.join(args.group_dir, 'hmol_figs')
    fig_dir = os.path.join(args.group_dir, 'figs')
    walk_dir = os.path.join(args.group_dir, 'walks')
    group_dir = os.path.join(args.group_dir, f'all_groups')
    label_dir = os.path.join(args.group_dir, f'all_groups/with_label')
    os.makedirs(hmol_fig_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(walk_dir, exist_ok=True)    
    os.makedirs(group_dir, exist_ok=True)   
    os.makedirs(label_dir, exist_ok=True)  

    data = []
    lines = open(args.data_file).readlines()
    data = [l.split(',')[0] for l in lines]    
    usage_count = defaultdict(int)
    segs = defaultdict(list)
    annotation_set = {}
    for i, line in enumerate(data):     
        print(f"segmenting mol {i+1}")   
                                    
        s = line.strip("\r\n ") 
        mol = Chem.MolFromSmarts(s)   
        Chem.Draw.MolToFile(mol, os.path.join(fig_dir, f'{i+1}.png'), size=(1000, 1000))      
        # Chem.Kekulize(mol, clearAromaticFlags=True)    
        annotated = False
        annotation_set[i+1] = fragment_mol(mol)
        if i+1 in annotation_set: 
            try:    
                segs[i+1] = process_annotation(mol, annotation_set[i+1])        
                annotated = True
            except Exception as e:           
                print(e)
                continue
        assert i+1 in segs                         
        # segs[i+1] = [','.join(map(str, range(1,mol.GetNumAtoms()+1)))]
        cur_len = len(vocab_mols)
        l, G = seg_mol(mol, segs[i+1], vocab_mols, l, i+1, annotated)                      
        if G is None:
            breakpoint()
            open(args.out_file, 'a+').write(f"{i+1}\n")
            continue
        else:
            for n in G:
                usage_count[n.split(':')[0]] += 1
        draw_hmol(os.path.join(hmol_fig_dir, f'{i+1}.png'), G)
        for _, mol, ind in vocab_mols[cur_len:]:  
            """
            A few special cases
            """
            for a in mol.GetAtoms():
                if a.GetSymbol() == 'N':
                    if sorted([n.GetSymbol() for n in a.GetNeighbors()]) == ['C','O','O']:                        
                        for b in a.GetBonds():
                            n = b.GetEndAtom()
                            if n.GetIdx() == a.GetIdx():
                                n = b.GetBeginAtom()                            
                            if n.GetSymbol() == 'O':
                                if mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1.0:
                                    n.SetFormalCharge(-1)
                                elif mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2.0:
                                    n.SetFormalCharge(0)                        
                        a.SetFormalCharge(1)                                
            for a in mol.GetAtoms():
                if a.GetSymbol() == 'N':
                    if [n.GetSymbol() for n in a.GetNeighbors()] == ['C','C','C','C']:                        
                        for b in a.GetBonds():
                            n = b.GetEndAtom()
                            if n.GetIdx() == a.GetIdx():
                                n = b.GetBeginAtom()                            
                            assert b.GetBondTypeAsDouble() == 1
                            n.SetFormalCharge(0)                        
                        a.SetFormalCharge(1)                                    
            if mol.GetNumAtoms() == 5 and len(mol.GetBonds()) == 4:
                for a in mol.GetAtoms(): # nitrate easter
                    if a.GetSymbol() == 'N':
                        if [n.GetSymbol() for n in a.GetNeighbors()] == ['O','O','O']:                        
                            for b in a.GetBonds():
                                n = b.GetEndAtom()
                                if n.GetIdx() == a.GetIdx():
                                    n = b.GetBeginAtom()
                                if n.GetSymbol() != 'O':
                                    breakpoint()
                                if b.GetBondTypeAsDouble() == 1.:
                                    if len(n.GetNeighbors()) == 2:                                
                                        n.SetFormalCharge(0)
                                    elif len(n.GetNeighbors()) == 1:
                                        n.SetFormalCharge(-1)
                                    else:
                                        breakpoint()
                                elif b.GetBondTypeAsDouble() == 2.:
                                    n.SetFormalCharge(0)                                               
                            a.SetFormalCharge(1)                        
            frags = Chem.GetMolFrags(mol) 
            if len(frags) != 1:
                print(f"mol {i+1} results in disconnected fragments!")
                break      
            try:
                Chem.rdmolfiles.MolToMolFile(mol, os.path.join(group_dir, f"{ind}.mol"))
            except:
                try:
                    Chem.SanitizeMol(mol)   
                    Chem.rdmolfiles.MolToMolFile(mol, os.path.join(group_dir, f"{ind}.mol"))
                except:
                    try:
                        Chem.SanitizeMol(mol)
                        Chem.rdmolfiles.MolToMolFile(mol, os.path.join(group_dir, f"{ind}.mol"))
                    except:
                        print("can't sanitize twice")
                        Chem.rdmolfiles.MolToMolFile(mol, os.path.join(group_dir, f"{ind}.mol"), kekulize=False)
            Draw.MolToFile(mol, os.path.join(label_dir, f'{ind}.png'), size=(100, 100))                       
        if not G.edges():
            assert len(G.nodes()) == 1
            root = list(G.nodes())[0]
            G.add_edge(root, root)

        nx.write_edgelist(G, os.path.join(walk_dir, f"walk_{i}.edgelist"))        
    
    sorted_order = [k[1:] for k in sorted(list(usage_count), key=lambda k: -usage_count[k])]
    pics = [os.path.join(label_dir, f"{ind}.png") for ind in sorted_order]
    titles = [usage_count[f"G{k}"] for k in sorted_order]
    titles = [f"{k}, usage: {v}" for k, v in zip(sorted_order, titles)]
    # draw_grid(pics, titles)
    return vocab_mols
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--group_dir')
    parser.add_argument('--data_file')    
    args = parser.parse_args()

    main(args)
