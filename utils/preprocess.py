import os
from rdkit import Chem
from rdkit.Chem import Draw
from collections import defaultdict
import re
import networkx as nx
from data import *

def name_group(m):
    return f"G{m}" # not group contrib
    prefix = lambda x: "P" if x <= 41 else ("S" if x <= 73 else "L")
    return prefix(m)+f"{m if m<=41 else m-41 if m<=73 else m-73}"



def load_mols(folder):
    mols = []
    files = os.listdir(folder)
    files = list(filter(lambda x: x.endswith('.mol'), files))
    files = sorted(files, key=lambda x:int(x.rstrip('.mol')))
    for f in files:
        if f.endswith('.mol'):
            mol = Chem.rdmolfiles.MolFromMolFile(os.path.join(folder, f))
            if not mol:
                mol = Chem.rdmolfiles.MolFromMolFile(os.path.join(folder, f), sanitize=False)
            assert mol
            mols.append(mol)
            for i, a in enumerate(mol.GetAtoms()): 
                a.SetProp("molAtomMapNumber", str(i+1))
        # these are the ones chem3d gets wrong
        # if f == '15.mol':
        #     mol = Chem.MolFromSmiles('CC1=CC(C)=CC=C1C')            
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))            
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/group-contrib/all_groups/15.mol')
        if 'group-contrib' in folder and f == '18.mol':
            mol = Chem.MolFromSmiles('C/C(C)=C(C)/C')
            for i, a in enumerate(mol.GetAtoms()): 
                a.SetProp("molAtomMapNumber", str(i+1))            
            Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/group-contrib/all_groups/18.mol')
            mols[-1] = mol
        if 'group-contrib' in folder and f == '3.mol':
            mol = Chem.MolFromSmiles('CC1=CC=C(C)C=C1')
            for i, a in enumerate(mol.GetAtoms()): 
                a.SetProp("molAtomMapNumber", str(i+1))            
            Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/group-contrib/all_groups/3.mol')
            mols[-1] = mol   
        if 'group-contrib' in folder and f == '11.mol':
            mol = Chem.MolFromSmiles('CC1=C(C)C=C(C)C=C1C')
            for i, a in enumerate(mol.GetAtoms()): 
                a.SetProp("molAtomMapNumber", str(i+1))            
            Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/group-contrib/all_groups/11.mol')
            mols[-1] = mol    
        if 'group-contrib' in folder and f == '13.mol':
            mol = Chem.MolFromSmiles('CC1=CC2=CC=C(C)C=C2C=C1')
            for i, a in enumerate(mol.GetAtoms()): 
                a.SetProp("molAtomMapNumber", str(i+1))            
            Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/group-contrib/all_groups/11.mol')
            mols[-1] = mol       
        if 'group-contrib' in folder and f == '14.mol':
            mol = Chem.MolFromSmiles('CC(C)=O')
            for i, a in enumerate(mol.GetAtoms()): 
                a.SetProp("molAtomMapNumber", str(i+1))            
            Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/group-contrib/all_groups/14.mol')
            mols[-1] = mol     
        # if 'permeability' in folder and f == '57.mol':
        #     wrong_inds = [8, 7]
        #     right_inds = [7, 8]
        #     inds = list(range(mols[-1].GetNumAtoms()))
        #     for w, r in zip(wrong_inds, right_inds):
        #         inds[w] = r
        #     mol = Chem.RenumberAtoms(mols[-1], inds)
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))                
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/datasetA_permeability/all_groups/57.mol')
        #     mols[-1] = mol
        # if 'permeability' in folder and f == '219.mol':
        #     wrong_inds = [2, 7, 6, 5, 4]
        #     right_inds = [4, 5, 6, 7, 2]
        #     inds = list(range(mols[-1].GetNumAtoms()))
        #     for w, r in zip(wrong_inds, right_inds):
        #         inds[w] = r
        #     mol = Chem.RenumberAtoms(mols[-1], inds)
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))                
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/datasetA_permeability/all_groups/219.mol')
        #     mols[-1] = mol            
        #     breakpoint()
        # if 'permeability' in folder and f == '222.mol':
        #     wrong_inds = [5,6,7,8,9,10]
        #     right_inds = [6,7,8,9,10,5]
        #     inds = list(range(mols[-1].GetNumAtoms()))
        #     for w, r in zip(wrong_inds, right_inds):
        #         inds[w] = r
        #     mol = Chem.RenumberAtoms(mols[-1], inds)
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))                
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/datasetA_permeability/all_groups/222.mol')
        #     mols[-1] = mol            
        #     breakpoint()
        # if 'permeability' in folder and f == '308.mol':
        #     fix_ring = (13, 18, 17, 16, 15, 14)
        #     mol = mols[-1]
        #     for i in range(len(fix_ring)):
        #         mol.GetBondBetweenAtoms(fix_ring[i], fix_ring[(i+1)%6]).SetBondType(Chem.rdchem.BondType.AROMATIC)
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))                
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/datasetA_permeability/all_groups/308.mol')
        #     mols[-1] = mol            
        #     breakpoint()
        # if 'permeability' in folder and f == '307.mol':
        #     fix_ring = [16,12,13,14,17,15]
        #     mol = mols[-1]
        #     for i in range(len(fix_ring)):
        #         mol.GetBondBetweenAtoms(fix_ring[i], fix_ring[(i+1)%6]).SetBondType(Chem.rdchem.BondType.AROMATIC)
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))                
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/datasetA_permeability/all_groups/307.mol')
        #     mols[-1] = mol            
        #     breakpoint()                        
        # if 'permeability' in folder and f == '307.mol':
        #     breakpoint()
        #     wrong_inds = [16,15,17]
        #     right_inds = [15,17,16]
        #     inds = list(range(mols[-1].GetNumAtoms()))
        #     for w, r in zip(wrong_inds, right_inds):
        #         inds[w] = r
        #     mol = Chem.RenumberAtoms(mols[-1], inds)
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))                
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/datasetA_permeability/all_groups/307.mol')
        #     mols[-1] = mol            
        #     breakpoint()   
        # if 'permeability' in folder and f == '396.mol':
            # breakpoint()
            # wrong_inds = [13,9,8]
            # right_inds = [9,8,13]
            # inds = list(range(mols[-1].GetNumAtoms()))
            # for w, r in zip(wrong_inds, right_inds):
            #     inds[w] = r
            # mol = Chem.RenumberAtoms(mols[-1], inds)
            # for i, a in enumerate(mol.GetAtoms()): 
            #     a.SetProp("molAtomMapNumber", str(i+1))                
            # Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/datasetA_permeability/all_groups/396.mol')
            # mols[-1] = mol            
            # breakpoint()               
        # if f == '47.mol':
        #     wrong_inds = [13,14,15,16]
        #     right_inds = [16,15,14,13]
        #     inds = list(range(mols[-1].GetNumAtoms()))
        #     for w, r in zip(wrong_inds, right_inds):
        #         inds[w] = r
        #     mol = Chem.RenumberAtoms(mols[-1], inds)
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))                
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/group-contrib/all_groups/47.mol')
        #     breakpoint()
        #     mols[-1] = mol

        # if f == '65.mol':
        #     wrong_inds = [13,14,15,16]
        #     right_inds = [16,15,14,13]
        #     inds = list(range(mols[-1].GetNumAtoms()))
        #     for w, r in zip(wrong_inds, right_inds):
        #         inds[w] = r
        #     mol = Chem.RenumberAtoms(mols[-1], inds)
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))                
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/group-contrib/all_groups/65.mol')
        #     breakpoint()
        #     mols[-1] = mol        
        # if f == '66.mol':
        #     wrong_inds = [13,14,15,16]
        #     right_inds = [16,15,14,13]
        #     inds = list(range(mols[-1].GetNumAtoms()))
        #     for w, r in zip(wrong_inds, right_inds):
        #         inds[w] = r
        #     mol = Chem.RenumberAtoms(mols[-1], inds)
        #     for i, a in enumerate(mol.GetAtoms()): 
        #         a.SetProp("molAtomMapNumber", str(i+1))                
        #     Chem.rdmolfiles.MolToMolFile(mol, 'data/datasets/group-contrib/all_groups/66.mol')
        #     breakpoint()        
    return mols


def load_walks(args):
    walks = []
    if os.path.isdir(args.data_file):
        for f in os.listdir(args.data_file):
            if f.endswith('.edgelist'):
                graph = nx.read_edgelist(os.path.join(args.data_file, f), create_using=nx.DiGraph)
                walks.append(graph)
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
