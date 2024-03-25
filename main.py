import argparse
import pickle
import random
import networkx as nx
from utils import *
from diffusion import L_grammar, Predictor, DiffusionGraph, DiffusionProcess, \
sample_walks, process_good_traj, get_repr, state_to_probs, append_traj, walk_edge_weight, load_dags, featurize_walk, do_predict, W_to_attr, attach_smiles, prune_walk
import json
import torch
import time
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
import pandas as pd
from tqdm import tqdm


class WalkDataset(Dataset):
    def __init__(self, 
                 args,
                 dags, 
                 procs,
                 props, 
                 graph, 
                 model, 
                 mol_feats, 
                 feat_lookup):
        self.args = args
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
        node_attr, edge_index, edge_attr = featurize_walk(self.args,
                                                          self.graph,
                                                          self.model, 
                                                          self.dags[idx], 
                                                          self.procs[idx][0], 
                                                          self.mol_feats, 
                                                          self.feat_lookup)                        
        edge_attr = edge_attr.detach()
        node_attr = node_attr.detach()
        return Data(edge_index=edge_index, x=node_attr, edge_attr=edge_attr, y=self.props[idx:idx+1], idx=idx)




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
    train_preds = [None for _ in train_order]
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
                node_attr, edge_index, edge_attr = featurize_walk(args, graph, model, dags_copy_train[ind], proc, mol_feats, feat_lookup)
            else:
                if ind not in train_feat_cache:
                    node_attr, edge_index, edge_attr = featurize_walk(args, graph, model, dags_copy_train[ind], proc, mol_feats, feat_lookup)
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
            train_preds[ind] = prop.detach().numpy()
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
    train_preds = np.stack(train_preds, axis=0)
    return train_loss_history, train_preds



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
                node_attr, edge_index, edge_attr = featurize_walk(args, graph, model, dags_copy_test[i], all_procs[i][0], mol_feats, feat_lookup)
            else:
                if i in test_feat_cache:
                    node_attr, edge_index, edge_attr = test_feat_cache[i]
                    assert node_attr.requires_grad == False
                else:
                    node_attr, edge_index, edge_attr = featurize_walk(args, graph, model, dags_copy_test[i], all_procs[i][0], mol_feats, feat_lookup)
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
    train_preds = []
    train_idxes = []
    times = []  
    for batch in tqdm(batch_loader):
        start_time = time.time()        
        opt.zero_grad()                 
        X, edge_index, edge_attr, props_y = batch.x, batch.edge_index, batch.edge_attr, batch.y
        train_idxes += batch.idx
        start_pred_time = time.time()                
        prop = do_predict(predictor, X, edge_index, edge_attr, batch=batch.batch, cuda=args.cuda)                                          
        if args.task == 'classification':
            prop = F.sigmoid(prop)         
        loss = loss_func(prop, props_y)        
        train_preds.append(prop.detach().numpy())
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
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_idx = np.zeros_like(train_preds)
    for i in range(train_preds.shape[0]):
        train_preds_idx[train_idxes[i]] = train_preds[i]
    return train_loss_history, train_preds_idx



def train(args, dags, graph, diffusion_args, props, norm_props, mol_feats, feat_lookup={}):
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
    all_procs_train = []
    all_procs_test = []
    # data augmentation
    for i in range(len(dags_copy_train)):
        proc = graph.lookup_process(dags_copy_train[i].dag_id)
        procs = [proc]
        if args.augment_dfs: # permutation of childs
            proc_total = proc.total
            if args.augment_order: # which node to start
                proc_total = proc.total * (len(proc.main_chain))
            for j in range(1, proc_total):
                procs.append(DiffusionProcess(dags_copy_train[i], graph.index_lookup, dfs_seed=j, **graph.diffusion_args))    
            if args.augment_dir:
                for j in range(-1, -proc_total, -1):
                    procs.append(DiffusionProcess(dags_copy_train[i], graph.index_lookup, dfs_seed=j, **graph.diffusion_args))
        # for proc in procs:
        #     print([a.id for a in proc.dfs_order])
        all_procs_train.append(procs)
    all_procs_test = []
    for i in range(len(dags_copy_test)):
        proc = graph.lookup_process(dags_copy_test[i].dag_id)
        procs = [proc]
        all_procs_test.append(procs)

    G = graph.graph  
    N = len(G)    
    diffusion_args['adj_matrix'] = nx.adjacency_matrix(G).toarray()
    if diffusion_args['e_init']:
        diffusion_args['init_e'] = nx.adjacency_matrix(G).toarray()    
    model = L_grammar(len(G), diffusion_args)
    state = torch.load(os.path.join(args.grammar_folder, 'ckpt.pt'))
    model.load_state_dict(state)

    # Per epoch        
        # For each walk i =: w_i
        # H_i <- EdgeConv(;H, w_i, E) (w_i DAG graph)
        # Optimize E, theta with Loss(MLP(H_i), prop-values)
    
    # init EdgeConv GNN
    predictor = Predictor(input_dim=args.input_dim, 
                          hidden_dim=args.hidden_dim, 
                          num_layers=args.num_layers, 
                          num_heads=len(norm_props[0]), 
                          gnn=args.gnn, 
                          edge_weights=args.edge_weights,
                          act=args.act, 
                          share_params=args.share_params,
                          in_mlp=args.in_mlp,
                          mlp_out=args.mlp_out,
                          dropout_rate=args.dropout_rate,
                          num_transformer_heads=args.num_transformer_heads,
                          init=args.init)
    if args.predictor_ckpt:
        state = torch.load(args.predictor_ckpt)
        predictor.load_state_dict(state, strict=True)
        # for p in predictor.parameters():
        #     p.requires_grad_(False)
    if args.cuda > -1:
        predictor.to(f"cuda:{args.cuda}")
    if args.update_grammar:
        all_params = []
        all_params.append({'params': list(model.parameters()), 'lr': args.grammar_lr})
        if not args.predictor_ckpt:
            all_params.append({'params': list(predictor.parameters()), 'lr': args.lr})
    else:
        all_params = list(predictor.parameters())

    if args.mol_feat == 'emb':
        emb = nn.Embedding(len(mols), 256)
        feat_lookup = {}
        for i in range(1,len(mols)+1):
            feat_lookup[name_group(i)] = emb.weight[i-1]
        mol_feats = torch.zeros((len(G), len(feat_lookup[list(feat_lookup)[0]])))        
        all_params += list(emb.parameters())
        for n in G.nodes():
            ind = graph.index_lookup[n]
            mol_feats[ind] = feat_lookup[n.split(':')[0]]                                   


    if args.opt == 'sgd':
        opt = torch.optim.SGD(all_params, lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        opt = torch.optim.Adam(all_params, lr=args.lr)
    else:
        raise
    if args.lr_schedule:
        scheduler = MultiStepLR(opt, milestones=args.lr_schedule, gamma=args.gamma)        
    
    loss_func = nn.MSELoss()   
    history = []
    train_history = []
    metrics = []
    best_loss = float("inf")
    print(args.logs_folder)
    mean_and_std = np.loadtxt(os.path.join(args.logs_folder,'mean_and_std.txt'))
    mean_and_std = np.atleast_2d(mean_and_std)
    mean = mean_and_std[:,0]
    std = mean_and_std[:,1]        

    if args.task == 'regression':
        best_maes = [float("inf") for _ in args.property_cols]
        best_r2s = [float("-inf") for _ in args.property_cols]
        best_avg_mae = float("inf")
        best_avg_r2 = float("-inf")
        best_maes_train = [float("inf") for _ in args.property_cols]
        best_r2s_train = [float("-inf") for _ in args.property_cols]
        best_avg_mae_train = float("inf")
        best_avg_r2_train = float("-inf")        
    else:
        best_accs = [float("-inf") for _ in args.property_cols]
        best_aucs = [float("-inf") for _ in args.property_cols]
        best_avg_acc = float("-inf")
        best_avg_auc = float("-inf")
        best_accs_train = [float("-inf") for _ in args.property_cols]
        best_aucs_train = [float("-inf") for _ in args.property_cols]
        best_avg_acc_train = float("-inf")
        best_avg_auc_train = float("-inf")        
    best_epochs = [0 for _ in args.property_cols]
    best_epochs_train = [0 for _ in args.property_cols]

    train_feat_cache = {}
    test_feat_cache = {}
    if args.batch_size == 1 and args.feat_cache and os.path.exists(args.feat_cache):
        train_feat_cache, test_feat_cache = pickle.load(open(args.feat_cache, 'rb'))        
    if args.batch_size > 1:
        train_dataset = WalkDataset(args,
                            dags_copy_train, 
                            all_procs_train,
                            norm_props_train,                           
                            graph, 
                            model, 
                            mol_feats, 
                            feat_lookup)
        test_dataset = WalkDataset(args,
                            dags_copy_test, 
                            all_procs_test,
                            norm_props_test,                           
                            graph, 
                            model, 
                            mol_feats, 
                            feat_lookup)    
        train_batch_loader = DataLoader(train_dataset, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers,
                                        prefetch_factor=args.prefetch_factor if args.prefetch_factor else None,
                                        shuffle=args.shuffle)
        eval_batch_loader = DataLoader(test_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    prefetch_factor=args.prefetch_factor if args.prefetch_factor else None)

    """
    MSELoss if regression
    CrossEntropy if classification
    """
    if 'PTC' in args.walks_file:    
        loss_func = nn.BCELoss()
    else:
        loss_func = nn.MSELoss(reduce=args.batch_size==1)
    for epoch in range(args.num_epochs):        
        # compute edge control weighted adj matrix via model inference    
        predictor.train()
        graph.reset()
        # random.shuffle(dags_copy_train)        
        train_order = list(range(len(dags_copy_train)))
        if args.shuffle:
            random.shuffle(train_order)                      

        if args.batch_size == 1:
            train_loss_history, train_preds = train_sgd(args, 
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
                                           all_procs_train)
        else:
            train_loss_history, train_preds = train_batch(args,
                                            opt,
                                            predictor,
                                            loss_func,
                                            train_batch_loader)

        graph.reset()
        predictor.eval()
        if args.batch_size == 1:
            res = eval_sgd(args,
                            model,
                            graph,
                            predictor,
                            loss_func,
                            mol_feats,
                            feat_lookup,
                            norm_props_test,
                            dags_copy_test,
                            test_feat_cache,
                            all_procs_test, 
                            mean, 
                            std)
            if args.plot_feats:
                test_preds, loss_history, test_feats = res
            else:
                test_preds, loss_history = res
        else:            
            test_preds, loss_history = eval_batch(args,
                                                predictor,
                                                loss_func,
                                                eval_batch_loader,
                                                mean,
                                                std)
        if args.lr_schedule:
            scheduler.step()            
        if args.batch_size == 1 and args.feat_cache and not os.path.exists(args.feat_cache):
            pickle.dump((train_feat_cache, test_feat_cache), open(args.feat_cache, 'wb+'))
        if np.mean(loss_history) < best_loss:
            best_loss = np.mean(loss_history)
            print(f"best_loss epoch {epoch}", best_loss)
            torch.save(predictor.state_dict(), os.path.join(args.logs_folder, f'predictor_ckpt_{best_loss}.pt'))
            if args.update_grammar:
                torch.save(model.state_dict(), os.path.join(args.logs_folder, f'grammar_ckpt_{best_loss}.pt'))
            if args.plot_feats:
                torch.save(test_feats, os.path.join(args.logs_folder, f'test_feats_{best_loss}.pt'))
            
        history.append(np.mean(loss_history))
        train_history.append(np.mean(train_loss_history))        
        y_hat = np.array(test_preds)
        y_hat_train = np.array(train_preds)        
        if args.norm_metrics:
            y = np.array(norm_props_test)
            y_train = np.array(norm_props_train)
        else:
            y = np.array(props_test)
            y_train = np.array(props_train)

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
        elif 'smiles_and_props.txt' in args.walks_file:
            col_names = ['H2','H2/N2'] 
            metric_names = [col_names[j] for j in args.property_cols]             
        elif 'smiles_and_props_old_O2_N2.txt' in args.walks_file:
            col_names = ['O2','O2/N2'] 
            metric_names = [col_names[j] for j in args.property_cols]                         
        elif 'smiles_and_props_old_CO2_CH4.txt' in args.walks_file:
            col_names = ['CO2','CO2/CH4'] 
            metric_names = [col_names[j] for j in args.property_cols]                                     
        else:
            print(f"assuming {args.walks_file} is group-contrib")
            col_names = ['H2','N2','O2','CH4','CO2']
            i1, i2 = args.property_cols
            metric_names = [f'permeability_{col_names[i1]}', f'selectivity_{col_names[i1]}_{col_names[i2]}']    
        metric = {}
        if not (y == y).all():
            breakpoint()
        if not (y_hat == y_hat).all():
            breakpoint()
        r2s = []
        accs = []
        aucs = []
        r2s_train = []
        accs_train = []
        aucs_train = []        
        for i in range(len(mean)):
            if args.task == 'regression':
                r2 = r2_score(y[:,i], y_hat[:,i])            
                r2s.append(r2)   
                mae = np.abs(y_hat[:,i]-y[:,i]).mean()
                mse = ((y_hat[:,i]-y[:,i])**2).mean()

                r2_train = r2_score(y_train[:,i], y_hat_train[:,i])            
                r2s_train.append(r2_train)                             
                mae_train = np.abs(y_hat_train[:,i]-y_train[:,i]).mean()
                mse_train = ((y_hat_train[:,i]-y_train[:,i])**2).mean()                

                if mae < best_maes[i]:
                    best_epochs[i] = len(metrics)
                    best_maes[i] = mae                
                    print(f"epoch {epoch} best {metric_names[i]} mae: {mae}")
                    best_epoch_r2 = r2
                    best_epoch_mae = mae
                    best_epoch_mse = mse
                else:
                    best_epoch_r2 = metrics[best_epochs[i]][f'{metric_names[i]}_r^2']
                    best_epoch_mae = metrics[best_epochs[i]][f'{metric_names[i]}_mae']
                    best_epoch_mse = metrics[best_epochs[i]][f'{metric_names[i]}_mse']

                if mae_train < best_maes_train[i]:
                    best_epochs_train[i] = len(metrics)
                    best_maes_train[i] = mae_train                
                    print(f"epoch {epoch} best {metric_names[i]} train mae: {mae_train}")
                    best_epoch_r2_train = r2_train
                    best_epoch_mae_train = mae_train
                    best_epoch_mse_train = mse_train
                else:
                    best_epoch_r2_train = metrics[best_epochs_train[i]][f'{metric_names[i]}_r^2_train']
                    best_epoch_mae_train = metrics[best_epochs_train[i]][f'{metric_names[i]}_mae_train']
                    best_epoch_mse_train = metrics[best_epochs_train[i]][f'{metric_names[i]}_mse_train']                    

                if r2 > best_r2s[i]:
                    best_r2s[i] = r2

                if r2_train > best_r2s_train[i]:
                    best_r2s_train[i] = r2_train                
            else:
                acc = ((y_hat[:,i]>0.5) == y[:,i]).mean()
                accs.append(acc)                
                auc = roc_auc_score(y[:,i],y_hat[:,i])
                aucs.append(auc)

                acc_train = ((y_hat_train[:,i]>0.5) == y_train[:,i]).mean()
                accs_train.append(acc_train)                    
                auc_train = roc_auc_score(y_train[:,i],y_hat_train[:,i])
                aucs_train.append(auc_train)       

                if acc > best_accs[i]:
                    best_epochs[i] = len(metrics)
                    best_accs[i] = acc
                    best_aucs[i] = auc
                    best_epoch_acc = acc
                    best_epoch_auc = auc
                    print(f"epoch {epoch} best {metric_names[i]} acc: {acc}")                    
                else:
                    best_epoch_acc = metrics[best_epochs[i]][f'{metric_names[i]}_acc']
                    best_epoch_auc = metrics[best_epochs[i]][f'{metric_names[i]}_auc']                 

                if acc_train > best_accs_train[i]:
                    best_epochs_train[i] = len(metrics)
                    best_accs_train[i] = acc_train
                    best_aucs_train[i] = auc_train
                    best_epoch_acc_train = acc_train
                    best_epoch_auc_train = auc_train
                    print(f"epoch {epoch} best {metric_names[i]} train acc: {acc_train}")                    
                else:
                    best_epoch_acc_train = metrics[best_epochs_train[i]][f'{metric_names[i]}_acc_train']
                    best_epoch_auc_train = metrics[best_epochs_train[i]][f'{metric_names[i]}_auc_train']                                     

            if args.task == 'regression':
                metric.update({
                    f'{metric_names[i]}_r^2': r2,
                    f'{metric_names[i]}_mae': mae,
                    f'{metric_names[i]}_mse': mse,
                })        
                metric.update({
                    f'best_{metric_names[i]}_r^2': best_epoch_r2,
                    f'best_{metric_names[i]}_mae': best_epoch_mae,
                    f'best_{metric_names[i]}_mse': best_epoch_mse,
                })  
                metric.update({
                    f'{metric_names[i]}_r^2_train': r2_train,
                    f'{metric_names[i]}_mae_train': mae_train,
                    f'{metric_names[i]}_mse_train': mse_train,
                })        
                metric.update({
                    f'best_{metric_names[i]}_r^2_train': best_epoch_r2_train,
                    f'best_{metric_names[i]}_mae_train': best_epoch_mae_train,
                    f'best_{metric_names[i]}_mse_train': best_epoch_mse_train,
                })                  
            else:
                metric.update({
                    f'{metric_names[i]}_acc': acc,
                    f'{metric_names[i]}_auc': auc
                })        
                metric.update({
                    f'best_{metric_names[i]}_acc': best_epoch_acc,
                    f'best_{metric_names[i]}_auc': best_epoch_auc,
                })  
                metric.update({
                    f'{metric_names[i]}_acc_train': acc_train,
                    f'{metric_names[i]}_auc_train': auc_train
                })        
                metric.update({
                    f'best_{metric_names[i]}_acc_train': best_epoch_acc_train,
                    f'best_{metric_names[i]}_auc_train': best_epoch_auc_train,
                })      

        if args.task == 'regression':
            avg_mae = np.abs(y_hat-y).mean()
            if avg_mae < best_avg_mae:
                best_avg_mae = avg_mae
            avg_r2 = np.mean(r2s)
            if avg_r2 > best_avg_r2:
                best_avg_r2 = avg_r2

            avg_mae_train = np.abs(y_hat_train-y_train).mean()
            if avg_mae_train < best_avg_mae_train:
                best_avg_mae_train = avg_mae_train
            avg_r2_train = np.mean(r2s_train)
            if avg_r2_train > best_avg_r2_train:
                best_avg_r2_train = avg_r2_train                
        else:
            avg_acc = ((y_hat>0.5) == y).mean()
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
            avg_auc = np.mean(aucs)
            if avg_auc > best_avg_auc:
                best_avg_auc = avg_auc    

            avg_acc_train = ((y_hat_train>0.5) == y_train).mean()
            if avg_acc_train > best_avg_acc_train:
                best_avg_acc_train = avg_acc_train
            avg_auc_train = np.mean(aucs_train)
            if avg_auc_train > best_avg_auc_train:
                best_avg_auc_train = avg_auc_train                          

        if args.task == 'regression':
            metric.update({"best_avg_mae": best_avg_mae})
            metric.update({"avg_mae": avg_mae})
            metric.update({"avg_best_mae": np.mean(best_maes)})
            metric.update({"avg_best_r2": np.mean(best_r2s)})
            print(f"epoch {epoch} avg best mae {np.mean(best_maes)}")        
            print(f"epoch {epoch} avg best r2 {np.mean(best_r2s)}")                

            metric.update({"best_avg_mae_train": best_avg_mae_train})
            metric.update({"avg_mae_train": avg_mae_train})
            metric.update({"avg_best_mae_train": np.mean(best_maes_train)})
            metric.update({"avg_best_r2_train": np.mean(best_r2s_train)})
            print(f"epoch {epoch} avg best train mae {np.mean(best_maes_train)}")        
            print(f"epoch {epoch} avg best train r2 {np.mean(best_r2s_train)}")                     
        else:
            metric.update({"best_avg_acc": best_avg_acc})
            metric.update({"avg_acc": avg_acc})
            metric.update({"best_avg_auc": best_avg_auc})
            metric.update({"avg_auc": avg_auc})            
            print(f"epoch {epoch} avg best acc {np.mean(best_accs)}")        
            print(f"epoch {epoch} avg best auc {np.mean(best_aucs)}")                          

            metric.update({"best_avg_acc_train": best_avg_acc_train})
            metric.update({"avg_acc_train": avg_acc_train})
            metric.update({"best_avg_auc_train": best_avg_auc_train})
            metric.update({"avg_auc_train": avg_auc_train})            
            print(f"epoch {epoch} avg best train acc {np.mean(best_accs_train)}")        
            print(f"epoch {epoch} avg best train auc {np.mean(best_aucs_train)}")               
                                      
        metrics.append(metric)
        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(args.logs_folder, 'metrics.csv'))
        fig_path = os.path.join(args.logs_folder, 'predictor_loss.png')
        fig = plt.Figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(len(history))+1, history, label='test loss')
        ax.plot(np.arange(len(train_history))+1, train_history, label='train loss')
        ax.text(0, min(history), "{:.3f}".format(min(history)))
        ax.text(0, min(train_history), "{:.3f}".format(min(train_history)))
        ax.axhline(y=min(history), label='min test loss', c='red')
        ax.axhline(y=min(train_history), label='min train loss', c='green')
        ax.set_title(f"Prediction loss")
        ax.set_ylim(ymin=0, ymax=5)
        ax.set_ylabel(f"MSE Loss")
        ax.set_xlabel('Epoch')
        ax.legend()
        fig.savefig(fig_path)  

    return model, predictor      


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
        elif 'smiles_and_props' in args.walks_file:
            prop = l.rstrip('\n').split()[1:]
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
            elif 'smiles_and_props' in args.walks_file:                            
                prop = list(map(float, prop))
                mask.append(i)
                props.append([prop[j] for j in args.property_cols])
                dags.append(dag_ids[i])                   
            else:
                try:
                    prop = list(map(lambda x: float(x) if x not in ['-','_'] else None, prop))
                except:
                    breakpoint()               
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
        norm_props = torch.FloatTensor((props-mean)/std)
    else:
        norm_props = torch.FloatTensor(props)
    if args.cuda > -1:
        norm_props = norm_props.to(f"cuda:{args.cuda}")
    return props, norm_props, dags, mask




def main(args):
    # Warmup epochs
        # Set E (edge weights), alpha as parameters
        # For each walk
            # Use DFS order to get S^(t), t=1,…
            # E(;S^(t-1)) <- E + f(context(S^(t-1)))
            # Construct L with E(;S^(t-1))
            # Optimize Loss(S^(t-1)+alpha*L(;S^(t-1)), S^(t))

    # in: graph G, n walks, m groups, F, f, E edge weights
    data, dags = load_dags(args)

    config_json = json.loads(json.load(open(os.path.join(args.grammar_folder,'config.json'),'r')))
    diffusion_args = {k[len('diffusion_'):]: v for (k, v) in config_json.items() if 'diffusion' in k}

    graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
    graph = DiffusionGraph(dags, graph, **diffusion_args)
    predefined_graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)    
    G = graph.graph
    N = len(G)   
    all_nodes = list(G.nodes())   

    if 'group-contrib' in args.motifs_folder:
        run_tests(graph, all_nodes)
 
    mols = load_mols(args.motifs_folder)
    red_grps = annotate_extra(mols, args.extra_label_path)    
    r_lookup = r_member_lookup(mols) 

    feat_lookup = {name_group(i): [] for i in range(1, len(mols)+1)}
    for mol_feat in args.mol_feat:
        if mol_feat == 'fp':            
            if ('permeability' in args.walks_file) or \
               ('crow' in args.walks_file) or \
                ('HOPV' in args.walks_file) or \
                ('PTC' in args.walks_file) or \
                ('lipophilicity' in args.walks_file) or \
                ('smiles_and_props' in args.walks_file):
                for i in range(1,len(mols)+1):
                    try:
                        feat_lookup[name_group(i)].append(mol2fp(mols[i-1]))    
                    except:   
                        print(f"cannot get fingerprint of {i}")  
                        # if 'lipophilicity' in args.walks_file and i == 166:
                        #     mols[i-1].GetAtomWithIdx(8).SetFormalCharge(1)
                        # elif 'lipophilicity' in args.walks_file and i == 421:
                        #     mols[i-1].GetAtomWithIdx(8).SetFormalCharge(1)
                        # elif 'lipophilicity' in args.walks_file and i == 509:
                        #     mols[i-1].GetAtomWithIdx(4).SetFormalCharge(1)
                        # elif 'lipophilicity' in args.walks_file and i == 563:
                        #     mols[i-1].GetAtomWithIdx(4).SetFormalCharge(1)
                        # elif 'lipophilicity' in args.walks_file and i == 937:
                        #     mols[i-1].GetAtomWithIdx(3).SetFormalCharge(1)
                        # elif 'lipophilicity' in args.walks_file and i == 1297:
                        #     mols[i-1].GetAtomWithIdx(1).SetFormalCharge(1)
                        # elif 'lipophilicity' in args.walks_file:
                        #     breakpoint()
                        if 'lipophilicity' in args.walks_file and i == 152:
                            mols[i-1].GetAtomWithIdx(8).SetFormalCharge(1)
                        elif 'lipophilicity' in args.walks_file and i == 395:
                            mols[i-1].GetAtomWithIdx(8).SetFormalCharge(1)
                        elif 'lipophilicity' in args.walks_file and i == 467:
                            mols[i-1].GetAtomWithIdx(4).SetFormalCharge(1)
                        elif 'lipophilicity' in args.walks_file and i == 869:
                            mols[i-1].GetAtomWithIdx(3).SetFormalCharge(1)
                        elif 'lipophilicity' in args.walks_file and i == 1189:
                            mols[i-1].GetAtomWithIdx(1).SetFormalCharge(1)
                        elif 'lipophilicity' in args.walks_file:
                            breakpoint()
                        if 'PTC'in args.walks_file and i == 350:
                            for a in mols[i-1].GetAtoms():
                                if a.GetSymbol() == 'N':
                                    a.SetFormalCharge(1)                                      
                        try:
                            mols[i-1].UpdatePropertyCache()                                    
                        except:
                            if 'smiles_and_props' in args.walks_file:
                                mols[i-1] = Chem.RWMol(mols[i-1])
                                # mols[i-1].RemoveAtom(0)
                                # mols[i-1].RemoveAtom(3)
                                for b in mols[i-1].GetBonds():
                                    if b.GetBondTypeAsDouble() == 3.:
                                        b.SetBondType(Chem.rdchem.BondType.SINGLE)
                                # mols[i-1].GetBondBetweenAtoms(9,10).SetBondType(Chem.rdchem.BondType.DOUBLE)
                                mols[i-1] = mols[i-1].GetMol()
                            else:
                                breakpoint()  
                        try:
                            mols[i-1].UpdatePropertyCache()  
                            FastFindRings(mols[i-1])
                            feat_lookup[name_group(i)].append(mol2fp(mols[i-1]))
                        except:    
                            print(f"failed getting mol feats of {i}")
                            feat_lookup[name_group(i)].append(np.zeros(2048,))
                       
            else:
                for i in range(1,98):
                    feat_lookup[name_group(i)].append(mol2fp(mols[i-1]))          
        elif mol_feat == 'emb':
            pass
        elif mol_feat == 'one_hot':            
            for i in range(1,len(mols)+1):
                feat = np.zeros((len(mols),))
                feat[i-1] = 1
                feat_lookup[name_group(i)].append(feat)
        elif mol_feat == 'ones':
            for i in range(1, len(mols)+1):
                feat_lookup[name_group(i)].append(np.ones(args.input_dim))
        elif mol_feat == 'W':
            for i in range(1, len(mols)+1):
                feat_lookup[name_group(i)].append(np.zeros(len(G)))
        elif mol_feat == 'unimol':
            assert len(mols) == 97
            unimol_feats = torch.load('data/group_reprs.pt')
            for i, unimol_feat in enumerate(unimol_feats):
                unimol_feats[i] = unimol_feat.mean(axis=0)
            unimol_feats = torch.stack(unimol_feats)            
            for i in range(1,98):
                feat_lookup[name_group(i)].append(unimol_feats[i-1])
        elif os.path.isdir(args.mol_feat_dir):
            sorted_files = sorted(os.listdir(args.mol_feat_dir), key=lambda x: int(Path(x).stem))
            files = [f for f in sorted_files if f.endswith('.npy')]
            feats = [np.load(os.path.join(args.mol_feat_dir, f)) for f in files]
            feats = np.stack(feats)            
            for i in range(1,len(mols)+1):
                feat_lookup[name_group(i)].append(feats[i-1])
        else:
            raise

    for k in feat_lookup:
        feat_lookup[k] = np.concatenate(feat_lookup[k])

    if args.mol_feat != 'emb':
        mol_feats = np.zeros((len(G), len(feat_lookup[list(feat_lookup)[0]])), dtype=np.float32)
        for n in G.nodes():
            ind = graph.index_lookup[n]
            mol_feats[ind] = feat_lookup[n.split(':')[0]]   
     

        input_dim = len(mol_feats[0])    
        if args.feat_concat_W: 
            input_dim += len(G)
    else:
        input_dim = 256
        mol_feats = torch.zeros((len(G), 256))
        feat_lookup = {}
        
    if args.concat_mol_feats:
        attach_smiles(args, dags)
        input_dim += 2048
    setattr(args, 'input_dim', input_dim)
    diffusion_args['adj_matrix'] = nx.adjacency_matrix(G).toarray()

    if args.predictor_file and args.grammar_file:    
        setattr(args, 'logs_folder', os.path.dirname(args.predictor_file))            
        model = L_grammar(N, diffusion_args)
        state = torch.load(args.grammar_file)
        model.load_state_dict(state)
        assert os.path.exists(os.path.join(args.logs_folder, 'config.json'))
        config = json.loads(json.load(open(os.path.join(args.logs_folder, 'config.json'))))
        predictor = Predictor(num_heads=len(args.property_cols), **config)
        state = torch.load(os.path.join(args.predictor_file))
        predictor.load_state_dict(state)  
        model.eval()
        predictor.eval()
        props, norm_props, dags, mask = preprocess_data(dags, args, os.path.dirname(args.predictor_file))      
    else:    
        predictor_path = os.path.join(args.grammar_folder,f'predictor_{time.time()}')
        os.makedirs(predictor_path, exist_ok=True)
        setattr(args, 'logs_folder', predictor_path)
        setattr(args, 'TORCH_SEED', int(os.environ.get('TORCH_SEED', '-1')))
        with open(os.path.join(predictor_path, 'config.json'), 'w+') as f:
            json.dump(json.dumps(args.__dict__), f)  

        props, norm_props, dags, mask = preprocess_data(dags, args, args.logs_folder)            
        model, predictor = train(args, dags, graph, diffusion_args, props, norm_props, mol_feats, feat_lookup)
        return
       
    graph.reset()
 
    seen_dags = deepcopy(dags)
    predict_args = {
        'predictor': predictor,
        'mol_feats': mol_feats,
        'feat_lookup': feat_lookup
    }
    novel, new_dags, trajs = sample_walks(args, G, graph, seen_dags, model, all_nodes, r_lookup, diffusion_args, predict_args=predict_args)
    if args.compute_metrics:
        metrics = compute_metrics(args, dags, new_dags)
        json.dump(metrics, open(os.path.join(args.logs_folder, 'main_metrics.json'), 'w+'))
    
    
    graph.reset()

    print("best novel samples")
    
    mean = [] ; std = []
    with open(os.path.join(args.logs_folder, 'mean_and_std.txt')) as f:
        while True:
            line = f.readline()
            if not line: break
            prop_mean, prop_std = map(float, line.split())
            mean.append(prop_mean)
            std.append(prop_std)
            
    out = []
    out_2 = []   
    if args.test_walks_file:
        fpath = os.path.join(args.logs_folder, 'novel_props_test_walks.txt') 
    else:
        fpath = os.path.join(args.logs_folder, 'novel_props.txt') 
    if 'permeability' in args.motifs_folder:
        col_names = ['log10_He_Bayesian','log10_H2_Bayesian','log10_O2_Bayesian','log10_N2_Bayesian','log10_CO2_Bayesian','log10_CH4_Bayesian']            
    elif 'group-contrib' in args.motifs_folder:
        col_names = ['H2','N2','O2','CH4','CO2']        
        assert len(args.property_cols) == 2
        i1, i2 = args.property_cols
        header = f"walk,{col_names[i1]},{col_names[i1]}/{col_names[i2]}\n"
        with open(fpath, 'w+') as f:
            f.write(header)
            for i, x in enumerate(novel):
                unnorm_prop = [x[-1][i]*std[i]+mean[i] for i in range(2)]
                out.append(unnorm_prop[0])
                out_2.append(unnorm_prop[1])
                novel[i][-1][0] = unnorm_prop[0]
                novel[i][-1][1] = unnorm_prop[1]
                f.write(f"{novel[i][0]},{unnorm_prop[0]},{unnorm_prop[1]}\n")
    else:
        if 'crow' in args.motifs_folder:
            col_names = ['tg_celsius']
        elif 'hopv' in args.motifs_folder:
            col_names = ['HOMO']
        elif 'lipophilicity' in args.motifs_folder:
            col_names = ['lipophilicity']
        else:
            raise NotImplementedError
        i1 = args.property_cols[0]        
        header = f"walk,{col_names[i1]}\n"
        with open(fpath, 'w+') as f:
            f.write(header)
            for i, x in enumerate(novel):
                unnorm_prop = [x[-1][i]*std[i]+mean[i] for i in range(1)]
                out.append(unnorm_prop[0])            
                novel[i][-1][0] = unnorm_prop[0]   
                f.write(f"{novel[i][0]},{unnorm_prop[0]}\n")                

    orig_preds = []   
    all_walks = {}
    all_walks['old'] = []
    for i, dag in enumerate(dags):
        proc = graph.lookup_process(dag.dag_id)
        node_attr, edge_index, edge_attr = featurize_walk(args, graph, model, dag, proc, mol_feats, feat_lookup)
        X = node_attr
        prop = do_predict(predictor, X, edge_index, edge_attr, cuda=args.cuda)           
        prop_npy = prop.detach().numpy()
        orig_preds.append(prop_npy)
        for j in range(len(proc.dfs_order)-1):
            a = proc.dfs_order[j]
            b = proc.dfs_order[j+1]
            # if not W_adj[graph.index_lookup[a.val]][graph.index_lookup[b.val]]:
            #     breakpoint()
        # get rid of cycle
        if 'group-contrib' in args.walks_file:
            conn = data[dag.dag_id][-1][:-1] # everything except last loop back
            W_adj = walk_edge_weight(dag, graph, model, proc)
        else:
            if dag.children:
                W_adj = walk_edge_weight(dag, graph, model, proc)
            else:
                W_adj = torch.zeros((N, N), dtype=torch.float32)
            conn = data[dag.dag_id][-1]    
            for (a, b, e) in conn:
                if e is None:
                    print(f"old dag {i} {a.val}-{b.val} is {e}")
        all_walks['old'].append((conn, W_adj, props[i]))                


    assert len(orig_preds[0]) == len(mean)
    orig_preds = [[orig_pred[i]*std[i]+mean[i] for i in range(len(orig_pred))] for orig_pred in orig_preds]
    if 'group-contrib'in args.walks_file:
        i1, i2 = args.property_cols            
        out, out_2, orig_preds = np.array(out), np.array(out_2), np.array(orig_preds)
        # props[:,1] = props[:,0]/props[:,1]
        # orig_preds[:,1] = orig_preds[:,0]/orig_preds[:,1]
        p1, p2 = np.concatenate((out,props[:,0])), np.concatenate((out_2,props[:,1]))    
        pareto_1, not_pareto_1, pareto_2, not_pareto_2 = pareto_or_not(p1, p2, len(out), min_better=False)
        fig = plt.Figure()    
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f'({col_names[i1]}, {col_names[i1]}/{col_names[i2]}) of original vs novel monomers')
        ax.scatter(out[not_pareto_1], out_2[not_pareto_1], c='b', label='predicted values of novel monomers')
        ax.scatter(out[pareto_1], out_2[pareto_1], c='b', marker='v')    
        ax.scatter(props[:,0][not_pareto_2], props[:,1][not_pareto_2], c='g', label='ground-truth values of original monomers')
        ax.scatter(props[:,0][pareto_2], props[:,1][pareto_2], c='g', marker='v')
        ax.scatter(orig_preds[:,0], orig_preds[:,1], c='r', label='predicted values of original monomers')
        ax.set_xlabel(f'Permeability {col_names[i1]}')
        ax.set_ylabel(f'Selectivity {col_names[i1]}/{col_names[i2]}')
        ax.set_ylim(ymin=0)
        ax.legend()
        ax.grid(True)    
        fig.savefig(os.path.join(args.logs_folder, 'pareto.png'))
    else:
        i1 = args.property_cols[0]
        orig_preds = [[orig_pred[i]*std[i]+mean[i] for i in range(1)] for orig_pred in orig_preds]
        out, orig_preds = np.array(out), np.array(orig_preds)     
        fig = plt.Figure()    
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f'({col_names[i1]}) of original vs novel molecules')
        ax.scatter(np.arange(len(out)), out, c='b', label='predicted values of novel molecules')
        ax.scatter(np.arange(len(props[:,0])), props[:,0], c='g', label='ground-truth values of original molecules')
        ax.scatter(np.arange(len(orig_preds[:,0])), orig_preds[:,0], c='r', label='predicted values of original molecules')
        ax.set_ylabel(f'{col_names[i1]}')
        ax.set_ylim(ymin=0)
        ax.legend()
        ax.grid(True)    
        fig.savefig(os.path.join(args.logs_folder, 'pareto.png'))        


        



    # for k, v in list(data_copy.items()):
    #     if k in mask:
    #         all_walks['old'].append(v)    
    
    with open(os.path.join(args.logs_folder, 'novel.txt'), 'w+') as f:
        for n in novel:
            f.write(n[0]+'\n')

    # if 'group-contrib' in args.walks_file:
    #     with open(os.path.join(args.logs_folder, 'novel_props.txt'), 'w+') as f:
    #         f.write(f"walk,{col_names[i1]},{col_names[i2]},{col_names[i1]}/{col_names[i2]},is_pareto\n")
    #         for i, x in enumerate(novel):
    #             assert len(novel[i][-1]) == 2
    #             print(f"{x[0]},{','.join(list(map(str,novel[i][-1][:2])))},{i in pareto_1}")
    #             f.write(f"{x[0]},{','.join(list(map(str,novel[i][-1][:2])))},{i in pareto_1}\n")
    # else:
    #     with open(os.path.join(args.logs_folder, 'novel_props.txt'), 'w+') as f:
    #         f.write(f"walk,{col_names[0]}\n")
    #         for i, x in enumerate(novel):
    #             assert len(novel[i][-1]) == len(args.property_cols)
    #             print(f"{x[0]},{','.join(list(map(str,novel[i][-1])))}")
    #             f.write(f"{x[0]},{','.join(list(map(str,novel[i][-1])))}\n")

    if 'group-contrib' in args.walks_file:
        for name_traj, root, edge_conn, W_adj, prop in novel: # (name_traj, root, edge_conn, W_adj, prop)
            assert not (edge_conn[-1][0].id and edge_conn[-1][1].id) # last edge is assumed to have root
        all_walks['novel'] = [(edge_conn[:-1], W_adj, prop) for x in novel] # all edges except last edge
    else:
        for name_traj, root, edge_conn, W_adj, prop in novel:
            all_walks['novel'] = [(edge_conn, W_adj, prop) for x in novel] # all edges except last edge

    # (edge, W_adj, prop) => ([(a,b,e,w)], prop)
    for key in ['old', 'novel']:
        all_walks[key] = prune_walk(args, graph, all_walks[key])
                      
    # pickle.dump(novel, open(os.path.join(args.logs_folder, 'novel.pkl', 'wb+')))
    breakpoint()
    all_walks['old'] = [[write_conn(x, G), *list(map(str, prop))] for x, prop in all_walks['old']]
    all_walks['novel'] = [[write_conn(x, G), *list(map(str, prop))] for x, prop in all_walks['novel']]
    print("novel", novel)
    json.dump(all_walks, open(os.path.join(args.logs_folder, 'all_dags.json'), 'w+'))
    print(f"done! {args.logs_folder}")


def run_tests(graph, all_nodes)                :
    # test append_traj
    traj, after = ['90', '50', '8'], 50
    assert append_traj(traj,after) == ['90', '50[->8]']
    traj, after = ['90', '50[->8]', '4', '25'], 4
    assert append_traj(traj,after) == ['90', '50[->8]', '4[->25]']
    traj, after = ['90', '50[->8]', '4[->25]'], 50
    assert append_traj(traj,after) == ['90', '50[->8,->4->25]']

    # test verify_walk    
    r_lookup = r_member_lookup(mols)     
    # verify_walk(r_lookup, graph, ['L3','S32','S20[->S1,->S1]','S32'], loop_back=True)
    # verify_walk(r_lookup, graph, ['L3','S32','S20[->P14->P39,->S18]','S32'], loop_back=True)
    
    # test process_good_traj
    name_traj = process_good_traj(['62','92','50[->39,->39]','92'], all_nodes)    
    name_traj = process_good_traj(['62','92','50[->12->37,->48]','92'], all_nodes)

    # test edge-connection guessing
    # verify_walk(r_lookup, graph.graph, ['S7','S3','S7'], loop_back=True)
    # verify_walk(r_lookup, graph.graph, ['S32','L14','S32'], loop_back=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motifs_folder')    
    parser.add_argument('--extra_label_path')    
    parser.add_argument('--grammar_folder')
    parser.add_argument('--grammar_file', help='if provided, sample new')
    parser.add_argument('--predictor_file', help='if provided, sample new')
    parser.add_argument('--predictor_ckpt')
    parser.add_argument('--update_grammar', action='store_true')
    parser.add_argument('--cuda', default=-1, type=int)

    # data params
    parser.add_argument('--all_dags_path', help='if given, do not sample, use this instead')    
    parser.add_argument('--predefined_graph_file')    
    parser.add_argument('--dags_file')
    parser.add_argument('--feat_cache', help='where to cache feats')
    parser.add_argument('--walks_file') 
    parser.add_argument('--test_walks_file', help='if given, sample these first') 
    parser.add_argument('--property_cols', type=int, default=[0,1], nargs='+', 
                        help='for group contrib, expect 2 cols of the respective permeabilities')
    parser.add_argument('--norm_metrics', action='store_true')
    parser.add_argument('--augment_dfs', action='store_true')
    parser.add_argument('--augment_order', action='store_true')
    parser.add_argument('--augment_dir', action='store_true')
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
    parser.add_argument('--mol_feat', type=str, default='W', nargs='+', choices=['fp', 
                                                                      'one_hot', 
                                                                      'ones', 
                                                                      'W', 
                                                                      'emb',
                                                                      'unimol', 
                                                                      'dir'])
    parser.add_argument('--concat_mol_feats', action='store_true')
    parser.add_argument('--mol_feat_dir', type=str)
    parser.add_argument('--feat_concat_W', action='store_true')
    parser.add_argument('--attn_W', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--edge_weights', action='store_true')
    parser.add_argument('--ablate_bidir', action='store_true')
    parser.add_argument('--gnn', default='gin')
    parser.add_argument('--act', default='relu')
    parser.add_argument('--share_params', action='store_false')
    parser.add_argument('--in_mlp', action='store_true')
    parser.add_argument('--mlp_out', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.)
    parser.add_argument('--num_transformer_heads', type=int, default=1)
    parser.add_argument('--init', default='normal', choices=['uniform', 'zeros', 'normal'])

    # sampling params
    parser.add_argument('--num_generate_samples', type=int, default=15)
    parser.add_argument('--compute_metrics', action='store_true', help='train and test metrics')
    parser.add_argument('--softmax', action='store_true')
    parser.add_argument('--max_thresh', type=float, default=0.9)
    parser.add_argument('--min_thresh', type=float, default=0.1)    
    # analysis and visualization params
    
    parser.add_argument('--vis_folder')
    parser.add_argument('--plot_feats', action='store_true')
    parser.add_argument('--vis_walk', action='store_true')

    args = parser.parse_args()    
    main(args)
