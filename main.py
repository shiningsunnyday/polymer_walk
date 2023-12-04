import argparse
import pickle
import random
import networkx as nx
from utils import *
from diffusion import L_grammar, Predictor, DiffusionGraph, DiffusionProcess, sample_walk, process_good_traj, get_repr, state_to_probs, append_traj
import json
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from tqdm import tqdm

def walk_edge_weight(dag, graph, model, proc):
    N = len(graph.graph)
    walk_order = []
    walk_order = proc.dfs_order
    context = torch.zeros((1, N), dtype=torch.float64)
    start_node_ind = graph.index_lookup[walk_order[0].val]
    prev_node_ind = start_node_ind
    W_adj = torch.zeros((N, N), dtype=torch.float32)
    t = 0
    state = torch.zeros((1, N), dtype=torch.float64)
    state[0, start_node_ind] = 1.
    for j in range(1, len(walk_order)):
        cur_node_ind = graph.index_lookup[walk_order[j%len(walk_order)].val]   
        # print(f"input state {get_repr(state)}, context {get_repr(context)}, t {t}")               
        update, context = model(state, context, t)
        state = state_to_probs(state+update)
        # print(f"post state {get_repr(state)}, context {get_repr(context)}, t {t}")  
        # dist = Categorical(state)
        # log_prob = dist.log_prob(cur_node_ind)
        t += 1
        W_adj[prev_node_ind, cur_node_ind] = state[0, cur_node_ind]
        W_adj[cur_node_ind, prev_node_ind] = state[0, cur_node_ind]
        # print(f"recounted {cur_node_ind} with prob {state[0, cur_node_ind]}")        
        state = torch.zeros((1, N), dtype=torch.float64)
        state[0, cur_node_ind] = 1.
        prev_node_ind = cur_node_ind  
    return W_adj 


def featurize_walk(graph, model, dag, proc, mol_feats, feat_lookup={}):
    """
    graph: DiffusionGraph
    model: L_grammar
    dag: Node
    proc: DiffusionProcess
    mol_feats: (len(graph.graph), dim) features of groups on graph.graph
    feat_lookup: features of isolated groups not on graph.graph
    """
    if dag.children:
        try:
            W_adj = walk_edge_weight(dag, graph, model, proc)
        except:
            breakpoint()
        # GNN with edge weight
        return W_to_attr(args, W_adj, mol_feats)
    else:
        assert feat_lookup, "need features for isolated groups"
        assert dag.val not in graph.graph
        assert len(proc.dfs_order) == 1
        N = len(graph.graph)
        W_adj = torch.zeros((N, N), dtype=torch.float32)
        mol_isolated_feats = feat_lookup[dag.val][None].repeat((N,1))
        node_attr, edge_index, edge_attr = W_to_attr(args, W_adj, mol_isolated_feats)
        assert edge_index.shape[1] == 0
        edge_index = torch.tensor([[0], [0]], dtype=torch.int64) # trivial self-connection for gnn
        return node_attr, edge_index, edge_attr
        


def train(args, dags, graph, diffusion_args, props, norm_props, mol_feats, feat_lookup={}):
    dags_copy = deepcopy(dags)
    if args.test_size:
        dags_copy_train, dags_copy_test, norm_props_train, norm_props_test, props_train, props_test, train_inds, test_inds = train_test_split(dags_copy, norm_props, props, list(range(len(dags_copy))), test_size=args.test_size, random_state=42)
    else:
        dags_copy_train, dags_copy_test, norm_props_train, norm_props_test, props_train, props_test, train_inds, test_inds = dags_copy, dags_copy, norm_props, norm_props, props, props, list(range(len(dags_copy))), list(range(len(dags_copy)))
    print(f"{len(dags_copy_train)} train dags, {len(dags_copy_test)} test dags")
    all_procs = []
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
        all_procs.append(procs)


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
                          act=args.act, 
                          share_params=args.share_params,
                          in_mlp=args.in_mlp,
                          mlp_out=args.mlp_out,
                          dropout_rate=args.dropout_rate)
    if args.predictor_ckpt:
        state = torch.load(args.predictor_ckpt)
        predictor.load_state_dict(state, strict=True)
        for p in predictor.parameters():
            p.requires_grad_(False)
    if args.cuda > -1:
        predictor.to(f"cuda:{args.cuda}")
    if args.update_grammar:
        all_params = []
        all_params.append({'params': list(model.parameters()), 'lr': args.grammar_lr})
        if not args.predictor_ckpt:
            all_params.append({'params': list(predictor.parameters()), 'lr': args.lr})
    else:
        all_params = list(predictor.parameters())


    if args.opt == 'sgd':
        opt = torch.optim.SGD(all_params, lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        opt = torch.optim.Adam(all_params, lr=args.lr)
    else:
        raise
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

    best_maes = [float("inf") for _ in args.property_cols]
    best_epochs = [0 for _ in args.property_cols]

    train_feat_cache = {}
    test_feat_cache = {}
    for epoch in range(args.num_epochs):        
        # compute edge control weighted adj matrix via model inference    
        predictor.train()
        graph.reset()
        # random.shuffle(dags_copy_train)
        train_loss_history = []
        train_order = list(range(len(dags_copy_train)))
        if args.shuffle:
            random.shuffle(train_order)
        
        times = []        
        for i in tqdm(range(len(dags_copy_train))):
            start_time = time.time()
            ind = train_order[i]
            procs = all_procs[ind]
            if i % args.num_accumulation_steps == 1 % args.num_accumulation_steps: 
                opt.zero_grad()         
            for proc in procs:
                start_feat_time = time.time()
                if args.update_grammar:
                    node_attr, edge_index, edge_attr = featurize_walk(graph, model, dags_copy_train[ind], proc, mol_feats, feat_lookup)
                else:
                    assert not args.shuffle
                    if ind not in train_feat_cache:
                        node_attr, edge_index, edge_attr = featurize_walk(graph, model, dags_copy_train[ind], proc, mol_feats, feat_lookup)
                        node_attr, edge_index, edge_attr = node_attr.detach(), edge_index.detach(), edge_attr.detach()
                        train_feat_cache[ind] = (node_attr, edge_index, edge_attr)                      
                    node_attr, edge_index, edge_attr = train_feat_cache[ind]
                    node_attr, edge_index, edge_attr = node_attr.clone(), edge_index.clone(), edge_attr.clone().detach()
                    assert node_attr.requires_grad == False
                
                X = node_attr        
                start_pred_time = time.time()                
                prop = do_predict(predictor, X, edge_index, edge_attr, cuda=args.cuda)                                          
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
            times.append([start_feat_time-start_time, start_pred_time-start_feat_time, loss_backward_time-start_pred_time, opt_step_time-loss_backward_time, final_time-opt_step_time])
        
        for time_type, time_took in zip(['zero grad', 'feat', 'pred', 'backward', 'step'], np.array(times).mean(axis=0).tolist()):
            print(f"{time_type} took {time_took}")


        graph.reset()
        test_preds = []
        loss_history = []
        predictor.eval()
        with torch.no_grad():
            for i in tqdm(range(len(dags_copy_test))):     
                if args.update_grammar:
                    node_attr, edge_index, edge_attr = featurize_walk(graph, model, dags_copy_test[i], graph.lookup_process(dags_copy_test[i].dag_id), mol_feats, feat_lookup)
                else:
                    if i in test_feat_cache:
                        node_attr, edge_index, edge_attr = test_feat_cache[i]
                        assert node_attr.requires_grad == False
                    else:
                        node_attr, edge_index, edge_attr = featurize_walk(graph, model, dags_copy_test[i], graph.lookup_process(dags_copy_test[i].dag_id), mol_feats, feat_lookup)
                        test_feat_cache[i] = (node_attr, edge_index, edge_attr)                
                X = node_attr
                prop = do_predict(predictor, X, edge_index, edge_attr, cuda=args.cuda)                          
                loss = loss_func(prop, norm_props_test[i])            
                loss_history.append(loss.item())
                test_preds.append((prop.cpu()*std+mean).numpy())



        if np.mean(loss_history) < best_loss:
            best_loss = np.mean(loss_history)
            print(f"best_loss epoch {epoch}", best_loss)
            torch.save(predictor.state_dict(), os.path.join(args.logs_folder, f'predictor_ckpt_{best_loss}.pt'))
            if args.update_grammar:
                torch.save(model.state_dict(), os.path.join(args.logs_folder, f'grammar_ckpt_{best_loss}.pt'))
            
        history.append(np.mean(loss_history))
        train_history.append(np.mean(train_loss_history))        
        y_hat = np.array(test_preds)
        y = np.array(props_test)

        if 'permeability' in args.walks_file:
            col_names = ['log10_He_Bayesian','log10_H2_Bayesian','log10_O2_Bayesian','log10_N2_Bayesian','log10_CO2_Bayesian','log10_CH4_Bayesian']            
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
        for i in range(len(mean)):
            r2 = r2_score(y[:,i], y_hat[:,i])
            mae = np.abs(y_hat[:,i]-y[:,i]).mean()
            mse = ((y_hat[:,i]-y[:,i])**2).mean()

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
        else:
            prop = l.rstrip('\n').split(' ')[-1]
            prop = prop.strip('(').rstrip(')').split(',')
     
        if args.property_cols:
            if 'permeability' in args.walks_file:
                prop = list(map(float, prop))
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
    
    norm_props = torch.FloatTensor((props-mean)/std)
    if args.cuda > -1:
        norm_props = norm_props.to(f"cuda:{args.cuda}")
    return props, norm_props, dags, mask


def do_predict(predictor, X, edge_index, edge_attr, cuda=-1):    
    if cuda > -1:
        X, edge_index, edge_attr = X.to(f"cuda:{cuda}"), edge_index.to(f"cuda:{cuda}"), edge_attr.to(f"cuda:{cuda}")
    # try modifying X based on edge_attr
    return predictor(X, edge_index, edge_attr)    


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
        if root_node.children:
          root_node.dag_id = k
          dags.append(root_node)

    config_json = json.loads(json.load(open(os.path.join(args.grammar_folder,'config.json'),'r')))
    diffusion_args = {k[len('diffusion_'):]: v for (k, v) in config_json.items() if 'diffusion' in k}

    graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
    graph = DiffusionGraph(dags, graph, **diffusion_args)
    predefined_graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)    
    G = graph.graph
    N = len(G)   
    all_nodes = list(G.nodes())   


    # run_tests(predefined_graph, all_nodes)
 
    mols = load_mols(args.motifs_folder)
    red_grps = annotate_extra(mols, args.extra_label_path)    
    r_lookup = r_member_lookup(mols) 

    if args.mol_feat == 'fp':
        feat_lookup = {}
        if 'permeability' in args.walks_file:
            for i in range(1,len(mols)+1):
                feat_lookup[name_group(i)] = mol2fp(mols[i-1])[0]            
        else:
            for i in range(1,98):
                feat_lookup[name_group(i)] = mol2fp(mols[i-1])[0]            
        mol_feats = np.zeros((len(G), 2048), dtype=np.float32)
        for n in G.nodes():
            ind = graph.index_lookup[n]
            mol_feats[ind] = feat_lookup[n.split(':')[0]]
    elif args.mol_feat == 'one_hot':
        mol_feats = np.eye(len(G), dtype=np.float32)
    elif args.mol_feat == 'ones':
        mol_feats = np.ones((len(G), args.input_dim), dtype=np.float32)
    elif args.mol_feat == 'W':
        mol_feats = torch.zeros((len(G), len(G)))
    elif args.mol_feat == 'unimol':
        assert len(mols) == 97
        unimol_feats = torch.load('data/group_reprs.pt')
        for i, unimol_feat in enumerate(unimol_feats):
            unimol_feats[i] = unimol_feat.mean(axis=0)
        unimol_feats = torch.stack(unimol_feats)
        feat_lookup = {}
        for i in range(1,98):
            feat_lookup[name_group(i)] = unimol_feats[i-1]
        mol_feats = np.zeros((len(G), 512), dtype=np.float32)
        for n in G.nodes():
            ind = graph.index_lookup[n]
            mol_feats[ind] = feat_lookup[n.split(':')[0]]
    elif os.path.isdir(args.mol_feat_dir):
        sorted_files = sorted(os.listdir(args.mol_feat_dir), key=lambda x: int(Path(x).stem))
        files = [f for f in sorted_files if f.endswith('.npy')]
        feats = [np.load(os.path.join(args.mol_feat_dir, f)) for f in files]
        feats = np.stack(feats)
        feats = torch.from_numpy(feats)
        feat_lookup = {}
        for i in range(1,len(mols)+1):
            feat_lookup[name_group(i)] = feats[i-1]
        mol_feats = np.zeros((len(G), feats.shape[-1]), dtype=np.float32)
        for n in G.nodes():
            ind = graph.index_lookup[n]
            mol_feats[ind] = feat_lookup[n.split(':')[0]]
    else:
        raise

    input_dim = len(mol_feats[0])
    if args.feat_concat_W: 
        input_dim += len(G)
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
        with open(os.path.join(predictor_path, 'config.json'), 'w+') as f:
            json.dump(json.dumps(args.__dict__), f)  

        props, norm_props, dags, mask = preprocess_data(dags, args, args.logs_folder)            
        model, predictor = train(args, dags, graph, diffusion_args, props, norm_props, mol_feats, feat_lookup)
       
    graph.reset()
    trajs = []
    novel = []
    lines = open(args.walks_file).readlines()
    walks = set()
    for i, l in enumerate(lines):        
        walk = l.rstrip('\n').split(' ')[-2] # -1 is prop val
        walks.add(walk)
    new_novel = 1
    seen_dags = deepcopy(dags)
    while len(novel) < args.num_generate_samples:
        print(f"add {new_novel} samples, now {len(novel)} novel samples")
        new_novel = 0
        for n in G.nodes():
            if ':' in n: continue
            traj, good = sample_walk(n, G, graph, model, all_nodes)                
            if len(traj) > 1 and good:                    
                name_traj = process_good_traj(traj, all_nodes)       
                assert len(traj) == len(name_traj)
                try:        
                    root, edge_conn = verify_walk(r_lookup, predefined_graph, name_traj)
                    DiffusionGraph.value_count(root, {}) # modifies edge_conn with :'s too
                    name_traj = '->'.join(name_traj)
                    trajs.append(name_traj)
                    # print(name_traj, "success")
                    if is_novel(seen_dags, root):
                        seen_dags.append(root)
                        print(name_traj, "novel")
                        walks.add(name_traj)                        
                        proc = DiffusionProcess(root, graph.index_lookup, **diffusion_args)
                        node_attr, edge_index, edge_attr = featurize_walk(graph, model, root, proc, mol_feats, feat_lookup)
                        W_adj = walk_edge_weight(root, graph, model, proc)
                        X = node_attr
                        prop = do_predict(predictor, X, edge_index, edge_attr, cuda=args.cuda)
                        print("predicted prop", prop)
                        probs = [W_adj[int(traj[i])][int(traj[(i+1)%len(traj)])] for i in range(len(traj))]
                        novel.append((name_traj, root, edge_conn, W_adj, prop.detach().numpy()))
                        new_novel += 1
                        # p (lambda W_adj,edge_conn,graph:[[a.id,a.val,b.id,b.val,e,W_adj[graph.index_lookup[a.val]][graph.index_lookup[b.val]].item(),W_adj[graph.index_lookup[b.val]][graph.index_lookup[a.val]].item()] for (a,b,e) in edge_conn])(W_adj,edge_conn,graph)                            
                    else:
                        print(f"{name_traj} discovered")
                except:
                    pass
            
    orig_preds = []   
    graph.reset()
    loss_history = []
    all_walks = {}
    all_walks['old'] = []
    for i, dag in enumerate(dags):
        proc = graph.lookup_process(dag.dag_id)
        node_attr, edge_index, edge_attr = featurize_walk(graph, model, dag, proc, mol_feats, feat_lookup)
        X = node_attr
        prop = do_predict(predictor, X, edge_index, edge_attr, cuda=args.cuda)           
        loss_history.append(nn.MSELoss()(prop, norm_props[i]).item())
        prop_npy = prop.detach().numpy()
        orig_preds.append(prop_npy)
        for j in range(len(proc.dfs_order)-1):
            a = proc.dfs_order[j]
            b = proc.dfs_order[j+1]
            # if not W_adj[graph.index_lookup[a.val]][graph.index_lookup[b.val]]:
            #     breakpoint()
        # get rid of cycle
        if not conn:
            breakpoint()
        conn = data[dag.dag_id][-1][:-1]
        W_adj = walk_edge_weight(dag, graph, model, proc)
        all_walks['old'].append((conn, W_adj, props[i]))

    print(np.mean(loss_history))

    print("best novel samples")
    mean = [] ; std = []
    with open(os.path.join(os.path.dirname(args.predictor_file), 'mean_and_std.txt')) as f:
        while True:
            line = f.readline()
            if not line: break
            prop_mean, prop_std = map(float, line.split())
            mean.append(prop_mean)
            std.append(prop_std)
            

    out = []
    out_2 = []
    i1, i2 = args.property_cols
    if 'permeability' in args.walks_file:
        col_names = ['log10_He_Bayesian','log10_H2_Bayesian','log10_O2_Bayesian','log10_N2_Bayesian','log10_CO2_Bayesian','log10_CH4_Bayesian']            
    else:
        print(f"assuming {args.walks_file} is group-contrib")
        col_names = ['H2','N2','O2','CH4','CO2']
        i1, i2 = args.property_cols
    with open(os.path.join(args.logs_folder, 'novel_props.txt'), 'w+') as f:
        f.write(f"walk,{col_names[i1]},{col_names[i2]},{col_names[i1]}/{col_names[i2]}\n")
        for i, x in enumerate(novel):
            unnorm_prop = [x[-1][i]*std[i]+mean[i] for i in range(2)]
            out.append(unnorm_prop[0])            
            out_2.append(unnorm_prop[1])
            novel[i][-1][0] = unnorm_prop[0]
            novel[i][-1][1] = unnorm_prop[1]
    
    orig_preds = [[orig_pred[i]*std[i]+mean[i] for i in range(2)] for orig_pred in orig_preds]
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
    

    write_conn = lambda conn: [(str(a.id), str(b.id), a.val.split(':')[0], b.val.split(':')[0], str(e), predefined_graph[a.val.split(':')[0]][b.val.split(':')[0]][e], str(w)) for (a,b,e, w) in conn]

    # for k, v in list(data_copy.items()):
    #     if k in mask:
    #         all_walks['old'].append(v)    
    
    with open(os.path.join(args.logs_folder, 'novel.txt'), 'w+') as f:
        for n in novel:
            f.write(n[0]+'\n')

    with open(os.path.join(args.logs_folder, 'novel_props.txt'), 'w+') as f:
        f.write(f"walk,{col_names[i1]},{col_names[i2]},{col_names[i1]}/{col_names[i2]},is_pareto\n")
        for i, x in enumerate(novel):
            print(f"{x[0]},{','.join(list(map(str,novel[i][-1][:2])))},{i in pareto_1}")
            f.write(f"{x[0]},{','.join(list(map(str,novel[i][-1][:2])))},{i in pareto_1}\n")

    for x in novel:
        assert not (x[2][-1][0].id and x[2][-1][1].id) # last edge is assumed to have root
    all_walks['novel'] = [(x[2][:-1], x[-2], x[-1]) for x in novel]  
    for key in ['old', 'novel']:
        for i in range(len(all_walks[key])):
            conn, W, prop = all_walks[key][i]
            pruned = []
            for j, edge in enumerate(conn):
                a, b, e = edge
                try:
                    w = W[graph.index_lookup[a.val]][graph.index_lookup[b.val]].item()
                except:
                    breakpoint()
                if key == 'novel' and w == 0.:
                    breakpoint()
                pruned.append((a, b, e, w))
            if not pruned:
                breakpoint()
            all_walks[key][i] = (pruned, prop)
     
                    
    # pickle.dump(novel, open(os.path.join(args.logs_folder, 'novel.pkl', 'wb+')))
    all_walks['old'] = [[write_conn(x), *list(map(str, prop))] for x, prop in all_walks['old']]
    all_walks['novel'] = [[write_conn(x), *list(map(str, prop))] for x, prop in all_walks['novel']]
    print("novel", novel)
    json.dump(all_walks, open(os.path.join(args.logs_folder, 'all_dags.json'), 'w+'))


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
    # verify_walk(r_lookup, graph, ['L3','S32','S20[->S1,->S1]','S32'])
    # verify_walk(r_lookup, graph, ['L3','S32','S20[->P14->P39,->S18]','S32'])
    
    # test process_good_traj
    name_traj = process_good_traj(['61','90','50[->39,->39]','90'], all_nodes)    
    assert name_traj == ['L3','S32','S20[->S1,->S1]','S32']
    name_traj = process_good_traj(['61','90','50[->12->37,->48]','90'], all_nodes)
    assert name_traj == ['L3','S32','S20[->P14->P39,->S18]','S32']




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
    parser.add_argument('--predefined_graph_file')    
    parser.add_argument('--dags_file')
    parser.add_argument('--walks_file') 
    parser.add_argument('--property_cols', type=int, default=[0,1], nargs='+', 
                        help='for group contrib, expect 2 cols of the respective permeabilities')
    parser.add_argument('--augment_dfs', action='store_true')
    parser.add_argument('--augment_order', action='store_true')
    parser.add_argument('--augment_dir', action='store_true')
    parser.add_argument('--test_size', default=0., type=float)

    # training params
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grammar_lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--num_accumulation_steps', type=int, default=1)
    parser.add_argument('--opt', default='adam')
    parser.add_argument('--shuffle', action='store_true')

    # model params
    parser.add_argument('--mol_feat', type=str, default='W', choices=['fp', 'one_hot', 'ones', 'W', 'unimol', 'dir'])
    parser.add_argument('--mol_feat_dir', type=str)
    parser.add_argument('--feat_concat_W', action='store_true')
    parser.add_argument('--attn_W', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--gnn', default='gin')
    parser.add_argument('--act', default='relu')
    parser.add_argument('--share_params', action='store_false')
    parser.add_argument('--in_mlp', action='store_true')
    parser.add_argument('--mlp_out', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.)

    # sampling params
    parser.add_argument('--num_generate_samples', type=int, default=100)

    args = parser.parse_args()
    
    main(args)
