from collections import OrderedDict, defaultdict
from copy import deepcopy
import numpy as np
from rdkit import Chem
import os


def name_group(m):
    """
    ANYTHING OTHER THAN GROUP CONTRIB
    """
    if 'group-contrib' in os.environ['dataset']:
        """GROUP CONTRIB"""
        prefix = lambda x: "P" if x <= 41 else ("S" if x <= 73 else "L")
        return prefix(m)+f"{m if m<=41 else m-41 if m<=73 else m-73}"
    else:
        return f"G{m}" # not group contrib


def join_mols(mol, other, r1, r2, folder=None):
    new_mol = Chem.CombineMols(mol, other)        
    sep = mol.GetNumAtoms()
    # duplicate inter-edges from mol's r1 to r2
    for (i, r_ind) in enumerate(r1):
        r = new_mol.GetAtomWithIdx(r_ind)
        ed_new = Chem.EditableMol(new_mol)
        for n in r.GetNeighbors():
            if n.GetIdx() in r1: continue
            cur_bond = new_mol.GetBondBetweenAtoms(r_ind, n.GetIdx())            
            ed_new.RemoveBond(r_ind, n.GetIdx())
            ed_new.AddBond(n.GetIdx(), r2[i]+sep, order=cur_bond.GetBondType())            
            new_mol = ed_new.GetMol()
            print(f"removed {(r_ind, n.GetIdx())}")
            print(f"added {(n.GetIdx(), r2[i]+sep)}")            
            if folder: 
                pass
                # print(create_stl(new_mol, folder))
    # delete mol's r1
    for r in sorted(r1, key=lambda x:-x):
        ed_new.RemoveAtom(r)
        print(f"removed {r}")
        if folder:     
            pass        
            # print(create_stl(new_mol, folder))        
    return ed_new.GetMol()


def walk_to_mol(walk, mols, folder=None):
    def extract_idxes(mol):
        dic = {}
        for i, a in enumerate(mol.GetAtoms()):
            prop = a.GetProp('a')
            for p in prop.split('_'):
                dic[p.split('-')[0], int(p.split('-')[1])] = i
        return dic

    idxes = {} # identify where atom is in the composed mol
    edges = defaultdict(list)   
    idx_mol = {}
    for v in walk:
        idx_mol[v[0]] = name_lookup[v[2]]-1
        idx_mol[v[1]] = name_lookup[v[3]]-1
    vis = {}
    for v in walk:
        vis[v[0]] = False
        vis[v[1]] = False
    for v in walk:
        edges[v[0]].append(v[1:])
    
    a = list(vis.keys())[0]        
    bfs = [a]
    vis[a] = True
    mol = deepcopy(mols[idx_mol[a]])
    for i, at in enumerate(mol.GetAtoms()):
        at.SetProp('a', f"{a}-{i}")
    
    idxes = extract_idxes(mol)
    while bfs:
        a = bfs[0]
        bfs.pop(0)
        breakpoint()
        for k in edges[a]:
            b, a_name, b_name, _, r_info = k[:5]     
            if vis[b]: 
                continue            
            r_info['k1'] = r_info['r_grp_1']+r_info['b1']
            r_info['k2'] = r_info['b2']+r_info['r_grp_2']
            mol_b = deepcopy(mols[idx_mol[b]])   
            for i, at in enumerate(mol_b.GetAtoms()):
                at.SetProp('a', f"{b}-{i}")
            a_node_inds = [idxes[(a, r)] for r in r_info['k1']]   
            b_node_inds = r_info['k2']
            # stl_path = create_stl(mol, folder=folder)[0]
            # print(f"{stl_path} starting molecule")
            mol = join_mols(mol, mol_b, a_node_inds, b_node_inds, folder=folder)
            # stl_path = create_stl(mol, folder=folder)[0]
            # print(f"joined {b_name} to {a_name} in {stl_path}")
            for a_ind, b_ind in zip(r_info['k1'], r_info['k2']):
                ind = mol.GetNumAtoms()-mol_b.GetNumAtoms()+b_ind
                prop = mol.GetAtomWithIdx(ind).GetProp('a')
                prop += f'_{a}-{a_ind}' # so we don't forget
                mol.GetAtomWithIdx(ind).SetProp('a', prop)
            idxes = extract_idxes(mol)
            vis[b] = True
            bfs.append(b)  

    return mol 


def extract_idxes(mol):
    """
    We want 
    """
    dic = {}
    for i, a in enumerate(mol.GetAtoms()):
        prop = a.GetProp('a')
        for p in prop.split('_'): # for each black atom matching "a as a red atom"
            # for each original group and its atom index
            dic[p.split('-')[0], int(p.split('-')[1])] = i
    return dic
def exact_match(mol1, mol2):
    mol1 = deepcopy(mol1)
    mol2 = deepcopy(mol2)
    for a in mol1.GetAtoms():
        a.SetAtomMapNum(0)
    for a in mol2.GetAtoms():
        a.SetAtomMapNum(0)            
    return Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2)



def walk_along_edge(mols,
         graph, 
         conn_index, 
         idx_val, 
         idx_mol, 
         idxes, 
         mol, 
         chosen_edge,
         new_mols, 
         new_chosen_edges, 
         new_idxes,
         a, b, e, 
         folder=None):
    r_info = graph[idx_val[a]][idx_val[b]][e]        
    r_info['k1'] = r_info['r_grp_1']+r_info['b1']
    r_info['k2'] = r_info['b2']+r_info['r_grp_2']
    # can we connect this way? we need to make sure atoms in r_info['r_grp_1']
    # have not been used before
    # so atom i in r_info['k1'] correspond to idxes[(a, i)]
    # then we can look into the atom property to see what's there
    used = False
    for j in r_info['r_grp_1']:
        prop = mol.GetAtomWithIdx(idxes[(a, j)]).GetProp('a')
        if '_' in prop:
            used = True
    if used:
        return
    mol_b = deepcopy(mols[idx_mol[b]])   
    for i, at in enumerate(mol_b.GetAtoms()):
        at.SetProp('a', f"{b}-{i}")
    # r_info tells us how to connect group a to b, so we need to find which atoms
    # that corresponds to in our composite mol
    a_node_inds = [idxes[(a, r)] for r in r_info['k1']]   
    b_node_inds = r_info['k2']
    # stl_path = create_stl(mol, folder=folder)[0]
    # print(f"{stl_path} starting molecule")
    enum_mol = deepcopy(mol)
    new_mol = join_mols(enum_mol, mol_b, a_node_inds, b_node_inds, folder=folder)

    # prune all the dups      
    exist = False
    for new_mol_prev in new_mols:
        if exact_match(new_mol_prev, new_mol):
            exist = True
    if exist:
        return

    new_mols.append(new_mol)
    new_chosen_edge = deepcopy(chosen_edge)
    new_chosen_edge[conn_index[(a,b)]] = e            
    new_chosen_edge[conn_index[(b,a)]] = e            
    new_chosen_edges.append(new_chosen_edge)
    # stl_path = create_stl(mol, folder=folder)[0]
    # print(f"joined {b_name} to {a_name} in {stl_path}")
    for a_ind, b_ind in zip(r_info['k1'], r_info['k2']):
        # compute the index of red atom in new mol
        ind = new_mol.GetNumAtoms()-mol_b.GetNumAtoms()+b_ind
        prop = new_mol.GetAtomWithIdx(ind).GetProp('a')
        # remember the original group and matching black atom
        prop += f'_{a}-{a_ind}' # so we don't forget
        new_mol.GetAtomWithIdx(ind).SetProp('a', prop)
    new_idx = extract_idxes(new_mol)
    new_idxes.append(new_idx)
    return True      



def walk_enumerate_mols(walk, graph, mols, folder=None, loop_back=False):
    name_lookup = {name_group(i): i for i in range(1, len(mols)+1)}
    # run some tests to make sure walk is the right format
    # last two edges of walk needs to be the same as first two
    if loop_back:
        assert walk[:2] == walk[-2:]
    idxes = {} # identify where atom is in the composed mol
    edges = defaultdict(list)   
    idx_mol = {}
    idx_val = {}
    conn_index = {}
    for v in walk:
        idx_mol[v[0]] = name_lookup[v[2]]-1
        idx_mol[v[1]] = name_lookup[v[3]]-1
        idx_val[v[0]] = v[2]
        idx_val[v[1]] = v[3]
    for ind, v in enumerate(walk):
        if (v[0],v[1]) not in conn_index:
            conn_index[(v[0],v[1])] = ind
    vis = {}
    for v in walk:
        vis[v[0]] = False
        vis[v[1]] = False
    for v in walk:
        edges[v[0]].append(v[1:]) # will be added twice for start and end group
    a = list(vis.keys())[0]        
    bfs = [a]
    vis[a] = True
    enum_mols = [deepcopy(mols[idx_mol[a]])]
    for i, at in enumerate(enum_mols[-1].GetAtoms()):
        at.SetProp('a', f"{a}-{i}") # this atom remembers it came from group a and was index i    
    idxes = [extract_idxes(enum_mols[-1])]
    chosen_edges = [[None for _ in walk]] # later use with conn
    while bfs:
        a = bfs[0]
        bfs.pop(0)                        
        for k in edges[a]:
            b, a_name, b_name = k                   
            if vis[b]:
                continue
            new_mols = []
            new_idxes = []     
            new_chosen_edges = []                  
            for idxes, mol, chosen_edge in zip(idxes, enum_mols, chosen_edges):
                for e in graph[idx_val[a]][idx_val[b]]:
                    walk_along_edge(mols, 
                                    graph, 
                                    conn_index, 
                                    idx_val, 
                                    idx_mol, 
                                    idxes, 
                                    mol, 
                                    chosen_edge, 
                                    new_mols, 
                                    new_chosen_edges, 
                                    new_idxes, 
                                    a, 
                                    b, 
                                    e, 
                                    folder)
                vis[b] = True
            bfs.append(b)  
            if new_mols:         
                enum_mols, idxes, chosen_edges = new_mols, new_idxes, new_chosen_edges
            else:
                raise  
    if loop_back:
        # loop back
        a, b = walk[-1][:2]
        # redefine conn_index
        conn_index = {}
        conn_index[(a,b)] = len(walk)-1
        conn_index[(b,a)] = len(walk)-2
        new_mols = []
        new_idxes = []     
        new_chosen_edges = []   
        for idxes, mol, chosen_edge in zip(idxes, enum_mols, chosen_edges):
            for i in graph[idx_val[a]][idx_val[b]]:
                walk_along_edge(mols, 
                                graph, 
                                conn_index, 
                                idx_val, 
                                idx_mol, 
                                idxes, 
                                mol, 
                                chosen_edge, 
                                new_mols, 
                                new_chosen_edges, 
                                new_idxes, 
                                a, 
                                b, 
                                i, 
                                folder)
        return new_chosen_edges[0], new_mols[0]
    else:
        return chosen_edges[0], new_mols[0]
