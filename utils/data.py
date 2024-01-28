import numpy as np 
from sklearn.metrics import r2_score
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
sys.path.append('/research/cbim/vast/zz500/Projects/mhg/ICML2024/my_data_efficient_grammar/')
sys.path.append('/home/msun415/my_data_efficient_grammar/')
import os
import fcntl
import time


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




def pareto_or_not(prop_1, prop_2, num_pred, min_better=True):
    pareto_1 = []
    not_pareto_1 = []
    pareto_2 = []
    not_pareto_2 = []
    for i, (v_1, v_2) in enumerate(zip(prop_1, prop_2)):
        if min_better and v_2 <= prop_2[np.argwhere(prop_1 <= v_1)].min():
            if i < num_pred: pareto_1.append(i)
            else: pareto_2.append(i-num_pred)
        elif not min_better and v_2 >= prop_2[np.argwhere(prop_1 >= v_1)].max():
            if i < num_pred: pareto_1.append(i)
            else: pareto_2.append(i-num_pred)            
        else:
            if i < num_pred: not_pareto_1.append(i)
            else: not_pareto_2.append(i-num_pred)
    return pareto_1, not_pareto_1, pareto_2, not_pareto_2

def chamfer_dist(old_dags, new_dags):
    div = InternalDiversity()
    dists = []
    for dag1 in old_dags:
        dist = []
        for dag2 in new_dags:
            mol1 = Chem.MolFromSmiles(dag1.smiles)
            mol2 = Chem.MolFromSmiles(dag2.smiles)
            dist.append(div.distance(mol1, mol2))
        dists.append(dist)
    dists = np.array(dists)
    d1 = dists[dists.argmin(axis=0), list(range(dists.shape[1]))].mean()
    d2 = dists[list(range(dists.shape[0])), dists.argmin(axis=1)].mean()
    return (d1+d2)/2


def retro_sender(generated_samples, folder):
    # File communication to obtain retro-synthesis rate
    sender_filename = os.path.join(folder, 'generated_samples.txt')
    receiver_filename = os.path.join(folder, 'output_syn.txt')
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
    return np.mean([int(eval(s[1])) for s in syn_status])




def compute_metrics(args, old_dags, new_dags):
    metrics = {}    
    if old_dags is not None:
        metrics['chamfer'] = chamfer_dist(old_dags, new_dags)
    mols = [Chem.MolFromSmiles(dag.smiles) for dag in new_dags]
    div = InternalDiversity()
    metrics['diversity'] = div.get_diversity(mols)
    # retro_res = [planner.plan(dag.smiles) for dag in new_dags]        
    # metrics['RS'] = retro_sender([dag.smiles for dag in new_dags], args.log_folder)
    return metrics 


def write_conn(conn, G):
    proc_conn = []
    for c in conn:
        if len(c) == 4:
            a,b,e,w = c
        else:
            a,b,e = c
        a_val = a.val.split(':')[0]
        b_val = b.val.split(':')[0]
        if e is not None:
            try:
                e = G[a_val][b_val][e]
            except:
                breakpoint()
        if len(c) == 4:
            proc_edge = (str(a.id), str(b.id), a_val, b_val, str(e), e, str(w))                 
        else:
            proc_edge = (str(a.id), str(b.id), a_val, b_val, str(e), e)                             
        proc_conn.append(proc_edge)
    return proc_conn

