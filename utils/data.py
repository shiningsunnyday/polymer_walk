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
import matplotlib.pyplot as plt

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


def plot_group_contrib(log_folder, out, out_2, orig_preds, props, col_names):
    assert len(col_names) == 2
    p1, p2 = np.concatenate((out,props[:,0])), np.concatenate((out_2,props[:,1]))
    pareto_1, not_pareto_1, pareto_2, not_pareto_2 = pareto_or_not(p1, p2, len(out), min_better=False)
    fig = plt.Figure()    
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'({col_names[0]}, {col_names[0]}/{col_names[1]}) of original vs novel monomers')
    ax.scatter(out[not_pareto_1], out_2[not_pareto_1], c='b', label='predicted values of novel monomers')
    ax.scatter(out[pareto_1], out_2[pareto_1], c='b', marker='v')    
    ax.scatter(props[:,0][not_pareto_2], props[:,1][not_pareto_2], c='g', label='ground-truth values of original monomers')
    ax.scatter(props[:,0][pareto_2], props[:,1][pareto_2], c='g', marker='v')
    ax.scatter(orig_preds[:,0], orig_preds[:,1], c='r', label='predicted values of original monomers')
    ax.set_xlabel(f'Permeability {col_names[0]}')
    ax.set_ylabel(f'Selectivity {col_names[0]}/{col_names[1]}')
    ax.set_ylim(ymin=0)
    ax.legend()
    ax.grid(True)    
    fig.savefig(os.path.join(log_folder, 'pareto.png'))
    return os.path.join(log_folder, 'pareto.png')


def plot_group_contrib_with_err(log_folder, all_outs, all_out_2s, all_orig_preds, all_props, col_names, err_thresh=[200, 2000], k=5.369*1e6, n=-2.636):
    out = np.stack(all_outs, axis=0)
    out_2 = np.stack(all_out_2s, axis=0)
    orig_preds = np.stack(all_orig_preds, axis=0)
    props = np.stack(all_props, axis=0)
    out = np.mean(all_outs, axis=0)
    out_2 = np.mean(all_out_2s, axis=0)
    orig_preds = np.mean(all_orig_preds, axis=0)
    props = np.mean(all_props, axis=0)
    out_std = np.std(all_outs, axis=0)
    out_2_std = np.std(all_out_2s, axis=0)
    orig_preds_std = np.std(all_orig_preds, axis=0)
    props_std = np.std(all_props, axis=0)
    
    out_str = ""
    out_2_str = ""
    for i in range(out.shape[0]):
        out_str += f"{out[i]} +- {out_std[i]}\n"
        out_2_str += f"{out_2[i]} +- {out_2_std[i]}\n"
    with open(os.path.join(log_folder, f"out_{col_names[0]}.txt"), 'w+') as f:
        f.write(out_str)
    with open(os.path.join(log_folder, f"out_{col_names[1]}.txt"), 'w+') as f:
        f.write(out_2_str)
    print(os.path.join(log_folder, f"out_{col_names[1]}.txt"))

    mask = out_std < err_thresh[0]
    mask_2 = out_2_std < err_thresh[1]
    mask = mask & mask_2
    out, out_2 = out[mask], out_2[mask]
    out_std, out_2_std = out_std[mask], out_2_std[mask]

    assert props_std.sum() < 1e-10
    assert len(col_names) == 2
    p1, p2 = np.concatenate((out,props[:,0])), np.concatenate((out_2,props[:,1]))
    pareto_1, not_pareto_1, pareto_2, not_pareto_2 = pareto_or_not(p1, p2, len(out), min_better=False)
    fig = plt.Figure()    
    ax = fig.add_subplot(1, 1, 1)

    sel_name = col_names[1].replace('_', '/')
    title = f'Permeability ({col_names[0]}) vs Selectivity ({sel_name})\nPerformance Plot of Original and Novel Monomers'
    ax.set_title(title, fontsize=16)
    ax.errorbar(out[not_pareto_1], out_2[not_pareto_1], xerr=out_std[not_pareto_1], yerr=out_2_std[not_pareto_1], c='b', label='predicted values of novel monomers\n(with std error bar over 3 seeds)', fmt='o')
    ax.errorbar(out[pareto_1], out_2[pareto_1], xerr=out_std[pareto_1], yerr=out_2_std[pareto_1], c='b', marker='v', fmt='o', label='novel monomers on the pareto front')    
    ax.scatter(props[:,0][not_pareto_2], props[:,1][not_pareto_2], c='g', label='ground-truth values of original monomers')
    ax.scatter(props[:,0][pareto_2], props[:,1][pareto_2], c='g', marker='v', label='original monomers on the pareto front')
    if k is not None and n is not None:
        x_min = props[:,0].min()
        x_max = props[:,0].max()
        x = np.linspace(x_min, x_max, 100)
        y = k**(-1/n) * x**(1/n)    
        ax.plot(x, y)
    # ax.errorbar(orig_preds[:,0], orig_preds[:,1], xerr=orig_preds_std[:,0], yerr=orig_preds_std[:,1], c='r', label='predicted values of original monomers', fmt='o')
    ax.scatter(orig_preds[:,0], orig_preds[:,1], c='r', label='predicted values of original monomers', marker='o')
    ax.set_xlabel(f'Permeability {col_names[0]}', fontsize=12)
    ax.set_ylabel(f'Selectivity {sel_name}', fontsize=12)         
    # from matplotlib.ticker import LogFormatter
    ax.set_yscale('log')
    # formatter = LogFormatter(labelOnlyBase=False)
    # ax.yaxis.set_major_formatter(formatter)
    if "O2/N2" in title:
        ax.set_yticks([3,4,6], minor=False)
        ax.set_yticklabels(["3","4","6"], minor=False)
        ax.set_yticks([5,7,8,9], minor=True)
    ax.set_xscale('log')    
    legend = ax.legend(loc='best', fontsize=8)
    figl = plt.Figure()
    ax_l = figl.add_subplot(1,1,1)
    ax_l.axis('off')
    figl.legend(handles=legend.legendHandles, labels=[text.get_text() for text in legend.get_texts()], loc='center')
    ax.grid(True)    
    fig.savefig(os.path.join(log_folder, 'pareto.pdf'))
    figl.savefig(os.path.join(log_folder, 'legend.pdf'))
    return os.path.join(log_folder, 'pareto.pdf'), mask



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

