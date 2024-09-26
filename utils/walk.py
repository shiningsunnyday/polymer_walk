from collections import OrderedDict, defaultdict
from copy import deepcopy
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
# rdDepictor.Compute2DCoords(new_mol)
# Chem.Draw.MolToImageFile(new_mol, '/home/msun415/new_mol.jpg',size=(2000,2000))
import os
from tqdm import tqdm

PRIORITY_RELATION = [('O', 'C')]


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
    

def vis_mol(mol, path):
    mol = deepcopy(mol)
    for i, a in enumerate(mol.GetAtoms()): 
        a.SetAtomMapNum(i)
        try:
            r = a.GetBoolProp('r')
        except:
            continue
        if r:
            a.SetProp('atomLabel', f"R_{a.GetSymbol()}{i+1}")
        else:
            a.SetProp('atomLabel', f"{a.GetSymbol()}:{i+1}")
    rdDepictor.Compute2DCoords(mol)
    Chem.Draw.MolToImageFile(mol, path, size=(2000,2000))
    return mol



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
            # use priority to determine if any atom symbol in r1 should be carried over
            sym_a = ed_new.GetMol().GetAtomWithIdx(r_ind).GetSymbol()
            no_a = ed_new.GetMol().GetAtomWithIdx(r_ind).GetAtomicNum()
            sym_b = ed_new.GetMol().GetAtomWithIdx(r2[i]+sep).GetSymbol() 
            ed_new.RemoveBond(r_ind, n.GetIdx())
            ed_new.AddBond(n.GetIdx(), r2[i]+sep, order=cur_bond.GetBondType())                        
            new_mol = ed_new.GetMol()
            if (sym_a, sym_b) in PRIORITY_RELATION:                         
                new_mol.GetAtomWithIdx(r2[i]+sep).SetAtomicNum(no_a)            
            ed_new = Chem.EditableMol(new_mol)
            # print(f"removed {(r_ind, n.GetIdx())}")
            # print(f"added {(n.GetIdx(), r2[i]+sep)}")            
            if folder: 
                pass
                # print(create_stl(new_mol, folder))    

    # delete mol's r1
    for r in sorted(r1, key=lambda x:-x):
        ed_new.RemoveAtom(r)
        if folder:     
            pass        
            # print(create_stl(new_mol, folder))
        
    # no explicit Hs allowed
    mol = ed_new.GetMol()
    for atom in mol.GetAtoms():
        atom.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(mol)
    except:
        # pass
        raise ValueError("cannot sanitize")
    return mol


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
        for k in edges[a]:
            b, a_name, b_name, _, r_info = k[:5]     
            if vis[b]: 
                continue  
            breakpoint()          
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
    if mol1.GetNumAtoms() != mol2.GetNumAtoms():
        return False
    if mol1.GetNumBonds() != mol2.GetNumBonds():
        return False    
    mol1 = deepcopy(mol1)
    mol2 = deepcopy(mol2)
    for a in mol1.GetAtoms():
        a.SetAtomMapNum(0)
    for a in mol2.GetAtoms():
        a.SetAtomMapNum(0)            
    return Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2)



def priority(mol_a, mol_b, a_inds, b_inds):
    # Check if there is an idx in a where its atomic symbol has higher precedence
    assert len(a_inds) == len(b_inds)
    for a, b in zip(a_inds, b_inds):
        a_sym = mol_a.GetAtomWithIdx(a).GetSymbol()
        b_sym = mol_b.GetAtomWithIdx(b).GetSymbol()
        if (a_sym, b_sym) in PRIORITY_RELATION:
            return True
    return False




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

    """
    The following function replaces a_node_inds in mol with b_node_inds in mol_b.
    In some cases, we want the opposite to be true. One way to disambiguate is to
    specify the priority assignment (a or b) in the syntax, but that makes notation
    too cumbersome. Instead, I use some heuristics, using the priority of atomic
    symbols (e.g. O > C). Specifically, I compare r_grp_1 and b2. If r_grp_1's atoms have
    higher priority b2, the joining is reversed, then the indices corrected.
    """

    # if priority(enum_mol, mol_b, [idxes[(a, at)] for at in r_info['r_grp_1']], r_info['b2']) \
    #     or priority(enum_mol, mol_b, [idxes[(a, at)] for at in r_info['b1']], r_info['r_grp_2']):
    #     ct_b = mol_b.GetNumAtoms()      
    #     ct_e = enum_mol.GetNumAtoms()     
    #     vis_mol(mol_b, '/home/msun415/mol_b.jpg')      
    #     vis_mol(enum_mol, '/home/msun415/enum_mol.jpg')      
    #     new_mol = join_mols(mol_b, enum_mol, b_node_inds, a_node_inds, folder=folder)                       
    #     old_mol = join_mols(enum_mol, mol_b, a_node_inds, b_node_inds, folder=folder)
    #     vis_mol(old_mol, '/home/msun415/old_mol.jpg')     
    #     inds = list(range(ct_e-len(a_node_inds), new_mol.GetNumAtoms()))+list(range(ct_e-len(a_node_inds)))           
    #     vis_mol(new_mol, '/home/msun415/new_mol.jpg')
    #     new_mol = Chem.RenumberAtoms(new_mol, inds)       
    #     vis_mol(new_mol, '/home/msun415/new_mol_renumbered.jpg')
    #     breakpoint()        
        
    # else:    
    new_mol = join_mols(enum_mol, mol_b, a_node_inds, b_node_inds, folder=folder)
    if Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)) is None:
        # pass
        breakpoint()

    # prune all the dups      
    exist = False
    # for new_mol_prev in tqdm(new_mols, "prune all dups"):
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



def walk_enumerate_mols(walk, graph, mols, folder=None, loop_back=False, return_all=False):
    name_lookup = {name_group(i): i for i in range(1, len(mols)+1)}
    # run some tests to make sure walk is the right format
    # make sure no dup edges    
    walk = list({(edge[0], edge[1]): edge for edge in walk}.values())
    idxes = {} # identify where atom is in the composed mol
    edges = defaultdict(list)   
    idx_mol = {}
    idx_val = {}
    
    for v in walk:
        idx_mol[v[0]] = name_lookup[v[2]]-1
        idx_mol[v[1]] = name_lookup[v[3]]-1
        idx_val[v[0]] = v[2]
        idx_val[v[1]] = v[3]
    conn_index = {}
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
            b, a_name, b_name, *pargs = k
            if vis[b]:
                continue
            new_mols = []
            new_idxes = []     
            new_chosen_edges = []                  
            for idxes, mol, chosen_edge in zip(idxes, enum_mols, chosen_edges):
                if len(pargs) == 1:
                    poss_edges = pargs
                elif len(pargs) == 0:
                    poss_edges = graph[idx_val[a]][idx_val[b]]
                else:
                    raise NotImplementedError
                bad = True
                for e in poss_edges:
                    try:
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
                        bad = False
                    except ValueError as e:
                        continue
                if bad: # cannot walk
                    raise ValueError("cannot sanitize mol")
                vis[b] = True
            bfs.append(b)  
            if new_mols:         
                enum_mols, idxes, chosen_edges = new_mols, new_idxes, new_chosen_edges
            else:
                raise KeyError("cannot walk along edge")
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
        chosen_edges = new_chosen_edges    
    if return_all:
        return chosen_edges, new_mols
    else:
        ind = 0
        for i in range(len(new_mols)-1,-1,-1):
            new_mol = new_mols[i]
            if Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)) is not None:
                ind = i        
        return chosen_edges[ind], new_mols[ind]