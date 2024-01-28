import argparse
import pickle
import random
import networkx as nx
from utils import *
from diffusion import L_grammar, Predictor, DiffusionGraph, DiffusionProcess, sample_walk, process_good_traj, get_repr, state_to_probs, append_traj, walk_edge_weight
import json
import torch
import time
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from xgboost import XGBRegressor
import pandas as pd
from tqdm import tqdm

name_to_idx = {name_group(i+1): i for i in range(len(mols))}


class WalkDataset(Dataset):
    def __init__(self, 
                 dags, 
                 procs,
                 props, 
                 graph, 
                 model, 
                 mol_feats, 
                 feat_lookup):
        self.props = props
        self.dags = dags
        self.procs = procs
        self.graph = graph
        self.model = model
        self.mol_feats = mol_feats
        self.feat_lookup = feat_lookup
        

    def __len__(self):
        return len(self.dags)


    def __getitem__(self, idx):
        assert len(self.procs[idx]) == 1
        node_attr, edge_index, edge_attr = featurize_walk(self.graph,
                                                          self.model, 
                                                          self.dags[idx], 
                                                          self.procs[idx][0], 
                                                          self.mol_feats, 
                                                          self.feat_lookup)                        
        edge_attr = edge_attr.detach()
        node_attr = node_attr.detach()
        return Data(edge_index=edge_index, x=node_attr, edge_attr=edge_attr, y=self.props[idx:idx+1])




def train_sgd(args, 
              opt, 
              model,
              graph,
              predictor,
              all_params,
              loss_func,
              mol_feats,
              feat_lookup,
              norm_props_train,             
              dags_copy_train, 
              train_feat_cache,
              train_order, 
              all_procs):
    train_loss_history = []
    times = []  
    for i in tqdm(range(len(dags_copy_train))):
        start_time = time.time()
        ind = train_order[i]
        procs = all_procs[ind]
        if i % args.num_accumulation_steps == 1 % args.num_accumulation_steps: 
            opt.zero_grad()  
        for proc in procs:
            start_feat_time = time.time()
            if args.update_grammar or args.mol_feat == 'emb':
                node_attr, edge_index, edge_attr = featurize_walk(graph, model, dags_copy_train[ind], proc, mol_feats, feat_lookup)
            else:
                if ind not in train_feat_cache:
                    node_attr, edge_index, edge_attr = featurize_walk(graph, model, dags_copy_train[ind], proc, mol_feats, feat_lookup)
                    node_attr, edge_index, edge_attr = node_attr.detach(), edge_index.detach(), edge_attr.detach()
                    train_feat_cache[ind] = (node_attr, edge_index, edge_attr)                      
                node_attr, edge_index, edge_attr = train_feat_cache[ind]
                node_attr, edge_index, edge_attr = node_attr.clone(), edge_index.clone(), edge_attr.clone().detach()
                assert node_attr.requires_grad == False
            X = node_attr        
            start_pred_time = time.time()                            
            if args.plot_feats:
                prop, feats = do_predict(predictor, X, edge_index, edge_attr, cuda=args.cuda, return_feats=args.plot_feats)                
            else:
                prop = do_predict(predictor, X, edge_index, edge_attr, cuda=args.cuda)
            if args.task == 'classification':
                prop = F.sigmoid(prop)            
            loss = loss_func(prop, norm_props_train[ind])
            train_loss_history.append(loss.item())     
            loss_backward_time = time.time()
            loss.backward()
            if not (loss == loss).all():
                breakpoint()                            
        if args.augment_dfs:
            assert args.num_accumulation_steps == 1
            for p in all_params:
                if p.requires_grad:
                    p.grad /= len(procs)
        opt_step_time = time.time()
        if i == len(dags_copy_train)-1 or i % args.num_accumulation_steps == 0:                                 
            opt.step()
        final_time = time.time()
        times.append([start_feat_time-start_time, 
                      start_pred_time-start_feat_time, 
                      loss_backward_time-start_pred_time, 
                      opt_step_time-loss_backward_time, 
                      final_time-opt_step_time])
    
    for time_type, time_took in zip(['zero grad', 'feat', 'pred', 'backward', 'step'], np.array(times).mean(axis=0).tolist()):
        print(f"{time_type} took {time_took}")    
    return train_loss_history



def eval_sgd(args, 
             model, 
             graph,
             predictor,             
             loss_func, 
             mol_feats, 
             feat_lookup, 
             norm_props_test,              
             dags_copy_test,  
             test_feat_cache, 
             all_procs,
             mean,
             std):             
    loss_history = []
    test_preds = []
    test_feats = []
    with torch.no_grad():
        for i in tqdm(range(len(dags_copy_test))):     
            assert len(all_procs[i]) == 1
            if args.update_grammar or args.mol_feat == 'emb':
                node_attr, edge_index, edge_attr = featurize_walk(graph, model, dags_copy_test[i], all_procs[i][0], mol_feats, feat_lookup)
            else:
                if i in test_feat_cache:
                    node_attr, edge_index, edge_attr = test_feat_cache[i]
                    assert node_attr.requires_grad == False
                else:
                    node_attr, edge_index, edge_attr = featurize_walk(graph, model, dags_copy_test[i], all_procs[i][0], mol_feats, feat_lookup)
                    test_feat_cache[i] = (node_attr, edge_index, edge_attr)                
            X = node_attr
            if args.plot_feats:
                prop, feats = do_predict(predictor, X, edge_index, edge_attr, cuda=args.cuda, return_feats=args.plot_feats)
            else:
                prop = do_predict(predictor, X, edge_index, edge_attr, cuda=args.cuda)
            if args.plot_feats:
                feats = torch.mean(feats, dim=0)
                test_feats.append(feats)
            if args.task == 'classification':
                prop = F.sigmoid(prop)                        
            loss = loss_func(prop, norm_props_test[i])            
            loss_history.append(loss.item())
            if args.norm_metrics:
                test_preds.append(prop.cpu().numpy())
            else:    
                test_preds.append((prop.cpu()*std+mean).numpy())    
    if args.plot_feats:
        test_feats = torch.stack(test_feats, dim=0)
        return test_preds, loss_history, test_feats
    else:
        return test_preds, loss_history



def eval_batch(args, 
               predictor,
               loss_func,
               batch_loader,
               mean,
               std):
    loss_history = np.empty((0, 1))
    test_preds = np.empty((0, mean.shape[-1]))
    with torch.no_grad():
        for batch in tqdm(batch_loader):
            X, edge_index, edge_attr, props_y = batch.x, batch.edge_index, batch.edge_attr, batch.y
            prop = do_predict(predictor, X, edge_index, edge_attr, batch=batch.batch, cuda=args.cuda)                          
            if args.task == 'classification':
                prop = F.sigmoid(prop)                
            loss = loss_func(prop, props_y)            
            loss_history = np.concatenate((loss_history, torch.atleast_2d(loss).numpy().mean(axis=-1, keepdims=True)), axis=0)
            if args.norm_metrics:
                test_preds = np.concatenate((test_preds, prop.cpu().numpy()), axis=0)
            else:
                test_preds = np.concatenate((test_preds, (prop.cpu()*std+mean).numpy()), axis=0)
    return test_preds, loss_history    



def train_batch(args,
                opt, 
                predictor,
                loss_func,
                batch_loader):                
    train_loss_history = np.empty((0, 1))
    times = []  
    for batch in tqdm(batch_loader):
        start_time = time.time()        
        opt.zero_grad()                 
        X, edge_index, edge_attr, props_y = batch.x, batch.edge_index, batch.edge_attr, batch.y
        start_pred_time = time.time()                
        prop = do_predict(predictor, X, edge_index, edge_attr, batch=batch.batch, cuda=args.cuda)                                          
        if args.task == 'classification':
            prop = F.sigmoid(prop)            
        loss = loss_func(prop, props_y)        
        train_loss_history = np.concatenate((train_loss_history, torch.atleast_2d(loss).detach().numpy().mean(axis=-1, keepdims=True)), axis=0)        
        loss_backward_time = time.time()
        loss.mean().backward()
        if not (loss == loss).all():
            breakpoint()        
        opt_step_time = time.time()                  
        opt.step()        
        final_time = time.time()
        times.append([start_pred_time-start_time, 
                      loss_backward_time-start_pred_time, 
                      opt_step_time-loss_backward_time, 
                      final_time-opt_step_time])    
    for time_type, time_took in zip(['zero grad', 'pred', 'backward', 'step'], np.array(times).mean(axis=0).tolist()):
        print(f"{time_type} took {time_took}")    
    return train_loss_history



def featurize_walk(graph, model, dag, proc, mol_feats, feat_lookup={}):
    """
    graph: DiffusionGraph
    model: L_grammar
    dag: Node
    proc: DiffusionProcess
    mol_feats: (len(graph.graph), dim) features of groups on graph.graph
    feat_lookup: features of isolated groups not on graph.graph
    """
    breakpoint()
    if dag.children:
        try:
            W_adj = walk_edge_weight(dag, graph, model, proc)
        except:
            breakpoint()
        # GNN with edge weight
        node_attr, edge_index, edge_attr = W_to_attr(args, W_adj, mol_feats)
    else:
        assert feat_lookup, "need features for isolated groups"
        assert dag.val not in graph.graph
        assert len(proc.dfs_order) == 1
        N = len(graph.graph)
        W_adj = torch.zeros((N, N), dtype=torch.float32)                
        if isinstance(feat_lookup[dag.val], torch.Tensor):
            feat = feat_lookup[dag.val][None]
            mol_isolated_feats = torch.tile(feat,[N,1])
        else:
            feat = feat_lookup[dag.val][None].astype('float32')
            mol_isolated_feats = np.tile(feat,[N,1])
        node_attr, edge_index, edge_attr = W_to_attr(args, W_adj, mol_isolated_feats)
        assert edge_index.shape[1] == 0
        edge_index = torch.tensor([[0], [0]], dtype=torch.int64) # trivial self-connection for gnn
        edge_attr = torch.tensor([[1.]])
    if hasattr(dag, 'smiles'):
        dag_mol = Chem.MolFromSmiles(dag.smiles)
        # if dag_mol is None: # try smarts
        #     dag_mol = Chem.MolFromSmarts(dag.smiles)        
        # else:
        #     dag_mol = Chem.AddHs(dag_mol)  
        # try:
        #     Chem.SanitizeMol(dag_mol)          
        #     if dag_mol is None:
        #         breakpoint()
        #     smiles_fp = torch.as_tensor(mol2fp(dag_mol), dtype=torch.float32)
        # except:
        #     smiles_fp = torch.zeros((2048,), dtype=torch.float32)
        smiles_fp = torch.as_tensor(mol2fp(dag_mol), dtype=torch.float32)
        node_attr = torch.concat((node_attr, torch.tile(smiles_fp, (node_attr.shape[0],1))), -1)
    return node_attr, edge_index, edge_attr
        


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


def regress_xgboost(smiles, y, test_idx, return_bst=False):

    def mol2fp(mol,nBits=1024):
        bitInfo={}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, bitInfo=bitInfo)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr, bitInfo


    trainmols = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        trainmols.append(mol)
    X = [mol2fp(mol)[0] for mol in trainmols]
    # X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
    # create model instance
    bst = XGBRegressor(n_estimators=16, max_depth=10, learning_rate=1)
    # fit model
    bst.fit([X[i] for i in range(len(smiles)) if i not in test_idx], [y[i] for i in range(len(smiles)) if i not in test_idx])
    # make predictions
    if return_bst: return bst.predict(X), bst
    else: return bst.predict(X)


def build_feat(dag):
    visited = {}
    def dfs(cur, feat, visited):
        visited[cur.id] = True
        if ':' in cur.val:
            breakpoint()
        index = name_to_idx[cur.val]
        feat[index] += 1
        for child, e in cur.children:
            if child.id in visited:
                continue
            dfs(child, feat, visited)
    feat = [0 for _ in name_to_idx]
    dfs(dag, feat, visited)
    return feat


def make_feats(args, dag):
    feat = build_feat(dag)
    if args.concat_mol_feats:
        dag_mol = Chem.MolFromSmiles(dag.smiles)                
        if dag_mol is None: # try smarts
            dag_mol = Chem.MolFromSmarts(dag.smiles)        
        else:
            dag_mol = Chem.AddHs(dag_mol)  
        try:
            Chem.SanitizeMol(dag_mol)          
            if dag_mol is None:
                breakpoint()
            smiles_fp = torch.as_tensor(mol2fp(dag_mol), dtype=torch.float32)
        except:
            smiles_fp = torch.zeros((2048,), dtype=torch.float32)            
        feat = np.concatenate((feat, smiles_fp), 0)    
    return feat



def train(args, dags, props, norm_props):    
    dags_copy = deepcopy(dags)
    if args.test_seed != -1:
        random.seed(args.test_seed)
        all_idx = list(range(len(dags_copy)))
        random.shuffle(all_idx)
        idx_train, idx_test = idx_partition(all_idx, all_idx, args.test_size, train_size=args.train_size)
        with open(os.path.join(args.logs_folder, 'test_idx.txt'), 'w+') as f:
            for idx in idx_test:
                f.write(f"{idx}\n")
        dags_copy_train, dags_copy_test = idx_partition(dags_copy, all_idx, args.test_size, train_size=args.train_size)
        norm_props_train, norm_props_test = idx_partition(norm_props, all_idx, args.test_size, train_size=args.train_size)
        props_train, props_test = idx_partition(props, all_idx, args.test_size, train_size=args.train_size)
        train_inds, test_inds = idx_partition(list(range(len(dags_copy))), all_idx, args.test_size, train_size=args.train_size)
    elif args.test_size:
        assert args.train_size == 1.
        dags_copy_train, dags_copy_test, norm_props_train, norm_props_test, props_train, props_test, train_inds, test_inds = train_test_split(dags_copy, norm_props, props, list(range(len(dags_copy))), test_size=args.test_size, random_state=42)
    else:
        assert args.train_size == 1.
        dags_copy_train, dags_copy_test, norm_props_train, norm_props_test, props_train, props_test, train_inds, test_inds = dags_copy, dags_copy, norm_props, norm_props, props, props, list(range(len(dags_copy))), list(range(len(dags_copy)))
    print(f"{len(dags_copy_train)} train dags, {len(dags_copy_test)} test dags")
    
    graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
    graph = DiffusionGraph(dags, graph)
    predefined_graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)    
    G = graph.graph
    N = len(G)   
    mean_and_std = np.loadtxt(os.path.join(args.logs_folder,'mean_and_std.txt'))
    mean_and_std = np.atleast_2d(mean_and_std)
    mean = mean_and_std[:,0]
    std = mean_and_std[:,1]      

    # Per epoch        
        # For each walk i =: w_i
        # H_i <- EdgeConv(;H, w_i, E) (w_i DAG graph)
        # Optimize E, theta with Loss(MLP(H_i), prop-values)
    
    # init xgboost
    predictor = XGBRegressor(n_estimators=16, max_depth=10, learning_rate=1)                                 
    X_train, y_train = [], []
    for i in tqdm(range(len(dags_copy_train))):
        feat = make_feats(args, dags_copy_train[i])
        X_train.append(feat)
        y_train.append(norm_props_train[i])
    X_test, y_test = [], []
    for i in tqdm(range(len(dags_copy_test))):
        feat = make_feats(args, dags_copy_test[i])
        X_test.append(feat)
        y_test.append(norm_props_test[i])       

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    predictor.fit(X_train, y_train)
    test_preds = predictor.predict(X_test)
    train_preds = predictor.predict(X_train)
    if len(test_preds.shape) == 1:
        test_preds = test_preds[:, None]
    if len(train_preds.shape) == 1:
        train_preds = train_preds[:, None]    
    if args.norm_metrics:
        y_test = np.array(norm_props_test)
        y_train = np.array(norm_props_train)
    else:
        test_preds = test_preds*std+mean
        train_preds = train_preds*std+mean
        y_test = np.array(props_test)
        y_train = np.array(props_train)
    y_hat_train = np.array(train_preds)    
    y_hat_test = np.array(test_preds)    
    if 'permeability' in args.walks_file:
        col_names = ['log10_He_Bayesian','log10_H2_Bayesian','log10_O2_Bayesian','log10_N2_Bayesian','log10_CO2_Bayesian','log10_CH4_Bayesian']            
        metric_names = [col_names[j] for j in args.property_cols]
    elif 'crow' in args.walks_file:
        col_names = ['tg_celsius']    
        metric_names = [col_names[j] for j in args.property_cols]                    
    elif 'HOPV' in args.walks_file:
        col_names = ['HOMO'] 
        metric_names = [col_names[j] for j in args.property_cols]  
    elif 'lipophilicity' in args.walks_file:
        col_names = ['lipophilicity'] 
        metric_names = [col_names[j] for j in args.property_cols]              
    elif 'PTC' in args.walks_file:
        col_names = ['carcinogenicity'] 
        metric_names = [col_names[j] for j in args.property_cols] 
    else:
        print(f"assuming {args.walks_file} is group-contrib")
        col_names = ['H2','N2','O2','CH4','CO2']
        i1, i2 = args.property_cols
        metric_names = [f'permeability_{col_names[i1]}', f'selectivity_{col_names[i1]}_{col_names[i2]}']    
    metric = {}
    metrics = []
    if not (y_test == y_test).all():
        breakpoint()
    if not (y_hat_test == y_hat_test).all():
        breakpoint()
           
    if args.task == 'regression':
        for i in range(len(metric_names)):
            prop = metric_names[i]
            r2 = r2_score(y_test[:,i], y_hat_test[:,i])            
            mae = np.abs(y_hat_test[:,i]-y_test[:,i]).mean()
            mse = ((y_hat_test[:,i]-y_test[:,i])**2).mean()                         
            metric.update({f"test_r2_{prop}": r2})
            metric.update({f"test_mae_{prop}": mae})
            metric.update({f"test_mse_{prop}": mse})             
            r2 = r2_score(y_train[:,i], y_hat_train[:,i])            
            mae = np.abs(y_hat_train[:,i]-y_train[:,i]).mean()
            mse = ((y_hat_train[:,i]-y_train[:,i])**2).mean()                         
            metric.update({f"train_r2_{prop}": r2})
            metric.update({f"train_mae_{prop}": mae})
            metric.update({f"train_mse_{prop}": mse})                         
    else:
        for i in range(len(metric_names)):
            prop = metric_names[i]
            acc = ((y_hat_test[:,i]>0.5) == y_test[:,i]).mean()
            auc = roc_auc_score(y_test[:,i],y_hat_test[:,i])            
            metric.update({f"test_acc_{prop}": acc})
            metric.update({f"test_auc_{prop}": auc})
            acc = ((y_hat_train[:,i]>0.5) == y_train[:,i]).mean()
            auc = roc_auc_score(y_train[:,i],y_hat_train[:,i])            
            metric.update({f"train_acc_{prop}": acc})
            metric.update({f"train_auc_{prop}": auc})            


    metrics.append(metric)
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(args.logs_folder, 'metrics.csv'))    
    print(os.path.join(args.logs_folder, 'metrics.csv'))


def preprocess_data(all_dags, args, logs_folder):
    lines = open(args.walks_file).readlines()
    props = []
    dag_ids = {}
    dags = []
    mask = []
    for dag in all_dags:
        dag_ids[dag.dag_id] = dag
    for i, l in enumerate(lines):        
        if i not in dag_ids: continue
        if 'permeability' in args.walks_file:            
            prop = l.rstrip('\n').split(',')[1:]
        elif 'crow' in args.walks_file:                 
            prop = l.rstrip('\n').split(',')[1:]
        elif 'HOPV' in args.walks_file:          
            prop = l.rstrip('\n').split(',')[1:]     
        elif 'lipophilicity' in args.walks_file:
            prop = l.rstrip('\n').split(',')[1:]
        elif 'polymer_walks' in args.walks_file:
            prop = l.rstrip('\n').split(' ')[-1]
            prop = prop.strip('(').rstrip(')').split(',')     
        elif 'PTC' in args.walks_file:            
            prop = l.rstrip('\n').split(',')[1:]
        else:
            breakpoint()

        if args.property_cols:
            if 'permeability' in args.walks_file:
                prop = list(map(float, prop))
                mask.append(i)
                props.append([prop[j] for j in args.property_cols])
                dags.append(dag_ids[i])
            elif 'crow' in args.walks_file or 'HOPV' in args.walks_file or 'lipophilicity' in args.walks_file:
                assert len(args.property_cols) == 1
                assert len(prop) == 1
                prop = list(map(float, prop))
                mask.append(i)
                props.append([prop[j] for j in args.property_cols])
                dags.append(dag_ids[i])     
            elif 'PTC' in args.walks_file:
                prop = list(map(int, prop))
                mask.append(i)
                props.append([prop[j] for j in args.property_cols])
                dags.append(dag_ids[i])                             
            else:
                try:
                    prop = list(map(lambda x: float(x) if x not in ['-','_'] else None, prop))
                except:
                    print(l)                
                i1, i2 = args.property_cols
                if prop[i1] and prop[i2]:     
                    mask.append(i)
                    props.append([prop[i1],prop[i1]/prop[i2]])
                    dags.append(dag_ids[i])
    props = np.array(props)
    mean, std = np.mean(props,axis=0,keepdims=True), np.std(props,axis=0,keepdims=True)    
    with open(os.path.join(logs_folder, 'mean_and_std.txt'), 'w+') as f:
        for i in range(props.shape[-1]):                
            f.write(f"{mean[0,i]} {std[0,i]}\n")
    
    if args.task == 'regression':
        norm_props = (props-mean)/std
    else:
        norm_props = props

    return props, norm_props, dags, mask


def do_predict(predictor, X, edge_index, edge_attr, batch=None, cuda=-1, return_feats=False):
    if cuda > -1:
        X, edge_index, edge_attr = X.to(f"cuda:{cuda}"), edge_index.to(f"cuda:{cuda}"), edge_attr.to(f"cuda:{cuda}")
    # try modifying X based on edge_attr
    if return_feats:
        out, feats = predictor(X, edge_index, edge_attr, return_feats=return_feats)
        if batch:
            breakpoint()
        else:
            node_mask = torch.unique(edge_index)
            feats = feats[node_mask] if predictor.share_params else [h[node_mask] for h in feats]
    y_hat = predictor(X, edge_index, edge_attr)
    if batch is None:
        node_mask = torch.unique(edge_index)
        y_hat = y_hat[node_mask]        
        out = y_hat.mean(axis=0)
    else:
        node_mask = torch.unique(edge_index)
        batch = batch[node_mask]
        y_hat = y_hat[node_mask]
        mean_aggr = aggr.MeanAggregation()        
        out = mean_aggr(y_hat, batch)
    if return_feats:
        return (out, feats)
    else:
        return out


def W_to_attr(args, W_adj, mol_feats):
    edge_index = W_adj.nonzero().T
    edge_attr = W_adj.flatten()[W_adj.flatten()>0][:, None]    
    if args.mol_feat == 'W':
        node_attr = W_adj
    else:
        node_attr = torch.as_tensor(mol_feats)
    if args.feat_concat_W:
        node_attr = torch.concat([node_attr, W_adj], dim=-1)
    return node_attr, edge_index, edge_attr



def attach_smiles(args, all_dags):
    lines = open(args.walks_file).readlines()
    dag_ids = {}
    for dag in all_dags:
        dag_ids[dag.dag_id] = dag
    if 'polymer_walks' in args.walks_file:
        assert hasattr(args, 'smiles_file')
        all_smiles = open(args.smiles_file).readlines()
        assert len(dag_ids) == len(all_smiles)
        polymer_smiles = {}
        for i, l in zip(dag_ids, all_smiles):
            if l == '\n':
                smiles = ''
            else:
                smiles = l.split(',')[0]
            polymer_smiles[i] = smiles
    for i, l in enumerate(lines):        
        if i not in dag_ids: continue
        if 'permeability' in args.walks_file:
            smiles = l.rstrip('\n').split(',')[0]
        elif 'crow' in args.walks_file or 'HOPV' in args.walks_file:
            smiles = l.rstrip('\n').split(',')[0]
        elif 'polymer_walks' in args.walks_file:
            if args.concat_mol_feats:
                smiles = polymer_smiles[i]
        elif 'PTC' in args.walks_file:
            smiles = l.rstrip('\n').split(',')[0]
        elif 'lipophilicity' in args.walks_file:
            smiles = l.rstrip('\n').split(',')[0]
        else:
            breakpoint()
        if args.concat_mol_feats:
            dag_ids[i].smiles = smiles   



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




def compute_metrics(args, old_dags, new_dags):
    metrics = {}    
    metrics['chamfer'] = chamfer_dist(old_dags, new_dags)
    mols = [Chem.MolFromSmiles(dag.smiles) for dag in new_dags]
    div = InternalDiversity()
    metrics['diversity'] = div.get_diversity(mols)
    # retro_res = [planner.plan(dag.smiles) for dag in new_dags]        
    # metrics['RS'] = sum([res is not None for res in retro_res])
    return metrics
    



def main(args):
    # Warmup epochs
        # Set E (edge weights), alpha as parameters
        # For each walk
            # Use DFS order to get S^(t), t=1,…
            # E(;S^(t-1)) <- E + f(context(S^(t-1)))
            # Construct L with E(;S^(t-1))
            # Optimize Loss(S^(t-1)+alpha*L(;S^(t-1)), S^(t))

    # in: graph G, n walks, m groups, F, f, E edge weights
    data = pickle.load(open(args.dags_file, 'rb'))    
    data_copy = deepcopy(data)
    dags = []
    for k, v in data.items():
        grps, root_node, conn = v
        # root_node, leaf_node, e = conn[-1]
        # assert root_node.id == 0
        # leaf_node.add_child((root_node, e)) # breaks dag
        # root_node.parent = (leaf_node, e)
        # if root_node.children:
        root_node.dag_id = k
        dags.append(root_node)


    predictor_path = os.path.join('./logs/', f'bag_of_motifs_{time.time()}')
    os.makedirs(predictor_path, exist_ok=True)
    setattr(args, 'logs_folder', predictor_path)         
    props, norm_props, dags, mask = preprocess_data(dags, args, args.logs_folder)
    if args.concat_mol_feats:
        attach_smiles(args, dags)       
    train(args, dags, props, norm_props)
    with open(os.path.join(predictor_path, 'config.json'), 'w+') as f:
        json.dump(json.dumps(args.__dict__), f)      
       






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motifs_folder')    
    parser.add_argument('--extra_label_path')   

    # data params
    parser.add_argument('--predefined_graph_file') 
    parser.add_argument('--dags_file')
    parser.add_argument('--smiles_file')
    parser.add_argument('--walks_file') 
    parser.add_argument('--property_cols', type=int, default=[0,1], nargs='+', 
                        help='for group contrib, expect 2 cols of the respective permeabilities')
    parser.add_argument('--norm_metrics', action='store_true')
    parser.add_argument('--test_seed', default=-1, type=int, help='seed for splitting data')
    parser.add_argument('--train_size', default=1., type=float)
    parser.add_argument('--test_size', default=0., type=float)
    parser.add_argument('--task', choices=['regression','classification'], default='regression')

    # training params
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--prefetch_factor', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_schedule', nargs='+', type=int, help="epochs to reduce lr")
    parser.add_argument('--grammar_lr', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--num_accumulation_steps', type=int, default=1)
    parser.add_argument('--opt', default='adam')
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--shuffle', action='store_true')

    # model params
    parser.add_argument('--concat_mol_feats', action='store_true')
    args = parser.parse_args()
    
    main(args)
