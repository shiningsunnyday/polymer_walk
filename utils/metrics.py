from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import fcntl
import numpy as np 
import time
import os
import shutil


def lock(f):
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        return False
    return True


class InternalDiversity():
    def distance(self, mol1, mol2, dtype="Tanimoto"):
        assert dtype in ["Tanimoto"]
        if dtype == "Tanimoto":
            sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2))
            return 1 - sim
        else:
            raise NotImplementedError

    def get_diversity(self, mol_list, dtype="Tanimoto"):
        similarity = 0
        mol_list = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mol_list] 
        for i in range(len(mol_list)):
            sims = DataStructs.BulkTanimotoSimilarity(mol_list[i], mol_list[:i])
            similarity += sum(sims)
        n = len(mol_list)
        n_pairs = n * (n - 1) / 2
        diversity = 1 - similarity / n_pairs
        return diversity
    


def chamfer_dist(old_smis, new_smis):
    div = InternalDiversity()
    dists = []
    for smi1 in old_smis:        
        dist = []
        for smi2 in new_smis:
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            if mol2 is None:
                continue
            print(f"smi1 {smi1} smi2 {smi2}")
            d = div.distance(mol1, mol2)
            dist.append(d)

        dists.append(dist)
    dists = np.array(dists)
    d1 = dists[dists.argmin(axis=0), list(range(dists.shape[1]))].mean()
    d2 = dists[list(range(dists.shape[0])), dists.argmin(axis=1)].mean()
    return (d1+d2)/2


def retro_sender(generated_samples, folder, retro_suffix=''):
    # File communication to obtain retro-synthesis rate
    sender_filename = os.path.join(folder, 'generated_samples.txt')
    receiver_filename = os.path.join(folder, 'output_syn.txt')
    open(sender_filename, 'w+').close() # clear
    open(receiver_filename, 'w+').close() # clear
    while(True):
        with open(sender_filename, 'r') as fr:
            editable = lock(fr)
            if editable:
                with open(sender_filename, 'w') as fw:
                    for sample in generated_samples:
                        fw.write('{}\n'.format(sample))
                break
            fcntl.flock(fr, fcntl.LOCK_UN)
    num_samples = len(generated_samples)
    print("Waiting for retro_star evaluation...")
    while(True):
        with open(receiver_filename, 'r') as fr:
            editable = lock(fr)
            if editable:
                syn_status = []
                lines = fr.readlines()
                if len(lines) == num_samples:
                    for idx, line in enumerate(lines):
                        splitted_line = line.strip().split()
                        syn_status.append((idx, splitted_line[2]))
                    break
            fcntl.flock(fr, fcntl.LOCK_UN)
        time.sleep(1)
    assert len(generated_samples) == len(syn_status)
    sender_filename_suffix = os.path.join(folder, f'generated_samples{retro_suffix}.txt')
    receiver_filename_suffix = os.path.join(folder, f'output_syn{retro_suffix}.txt')
    shutil.copyfile(sender_filename, sender_filename_suffix) # save res
    shutil.copyfile(receiver_filename, receiver_filename_suffix) # save res
    return np.mean([int(eval(s[1])) for s in syn_status])


def get_novelty(old_smiles, new_smiles):
    old_smiles = [Chem.CanonSmiles(smi) for smi in old_smiles]
    res = []
    for smi in new_smiles:  
        try:
            smi = Chem.CanonSmiles(smi)    
        except:
            continue
        res.append(smi not in old_smiles)
    return sum(res)/len(res)


def get_uniqueness(smis):
    canon = set()
    uniq = 0
    for smi in smis:
        try:
            smi = Chem.CanonSmiles(smi)
        except:
            continue
        if smi not in canon:
            canon.add(smi)
            uniq += 1
    return uniq/len(smis)


def data_driven_membership(mols, old_dags, new_smiles):
    def dfs_count(dag, count):
        count[dag.val] = count.get(dag.val, 0) + 1
        for c in dag.children:
            if c[0].id == 0: # cyclic
                continue
            dfs_count(c[0], count)    
    # data driven way
    occur = {}
    all_counts = {}
    for dag in old_dags:
        count = {}
        dfs_count(dag, count)
        for k in count:
            occur[k] = occur.get(k, 0) + 1
            all_counts[k] = all_counts.get(k, 0) + count[k]
    freq = sorted(occur, key=lambda k: occur[k])
    common = freq[-1]
    assert 'G' in common        
    motif = mols[int(common.split('G')[-1])-1]    
    member = 0
    for smiles in new_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol.GetSubstructMatches(motif):
            member += 1
    return member/len(new_smiles)   


def check_hopv(smis):
    thio = Chem.MolFromSmiles('[cH:1]1[cH:2][cH:3][cH:4][s:5]1') # thiophene
    members = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        members += bool(mol.GetSubstructMatches(thio))
    return members/len(smis)


def check_ptc(smis):    
    halide_smis = ['[CH2:1]([Br:2])[CH3:3]', '[Cl:1][CH3:2]']
    halides = [Chem.MolFromSmiles(s) for s in halide_smis]
    members = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        members += max(*[bool(mol.GetSubstructMatches(halide)) for halide in halides])
    return members/len(smis)


def get_membership(args, new_smiles):    
    if 'hopv' in args.data_file:
        func = check_hopv
    elif 'ptc' in args.data_file:
        func = check_ptc
    else:
        breakpoint()  
    return func(new_smiles)    



def compute_metrics(args, mols, old_smiles, new_smiles, retro_suffix='_test'):
    metrics = {}    
    metrics['membership'] = get_membership(args, new_smiles)
    mols = [Chem.MolFromSmiles(smi) for smi in new_smiles]
    breakpoint()
    metrics['valid'] = sum([mol is not None for mol in mols])/len(mols)     
    if old_smiles is not None:        
        metrics['novelty'] = get_novelty(old_smiles, new_smiles)
        metrics['chamfer'] = chamfer_dist(old_smiles, new_smiles)       
    metrics['unique'] = get_uniqueness(new_smiles)
    div = InternalDiversity()
    mols = [mol for mol, smi in zip(mols, new_smiles) if mol is not None]
    new_smiles = [smi for smi in new_smiles if Chem.MolFromSmiles(smi) is not None]
    print(f"{len(mols)} valid")
    metrics['diversity'] = div.get_diversity(mols)
    # retro_res = [planner.plan(dag.smiles) for dag in new_dags]        
    log_folder = args.log_folder if hasattr(args, 'log_folder') else args.logs_folder
    metrics['RS'] = retro_sender(new_smiles, log_folder, retro_suffix=retro_suffix)
    return metrics 
