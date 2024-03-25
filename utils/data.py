import numpy as np 
from sklearn.metrics import r2_score
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
sys.path.append('/research/cbim/vast/zz500/Projects/mhg/ICML2024/my_data_efficient_grammar/')
sys.path.append('/home/msun415/my_data_efficient_grammar/')
import os
import json
import torch

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


def idx_partition(data, all_idx, test_size=0.2, train_size=0.8):
    train_size = min(train_size, 1-test_size)
    assert len(data) == len(all_idx)
    train_mask = all_idx[:int(train_size*len(data))]
    test_mask = all_idx[int((1-test_size)*len(data)):]
    train, test = [data[i] for i in train_mask], [data[i] for i in test_mask]
    if isinstance(data, torch.Tensor):
        train = torch.stack(train)
        test = torch.stack(test)
    return train, test


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


def json_loads(s):
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except:
        return json.loads(s.replace("'", '"'))

