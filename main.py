import argparse
import pickle
import random
import networkx as nx
from utils.graph import *
from diffusion import L_grammar, Predictor, DiffusionGraph, DiffusionProcess, sample_walk, process_good_traj
import json
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split

def walk_edge_weight(dag, graph, model, proc):
    N = len(graph.graph)
    walk_order = []
    assert not proc.split
    walk_order = proc.dfs_order
    context = torch.zeros((1, N), dtype=torch.float64)
    start_node_ind = graph.index_lookup[walk_order[0].val]
    prev_node_ind = start_node_ind
    context[0, start_node_ind] = 1.            
    W_adj = torch.zeros((N, N), dtype=torch.float32)
    t = 0
    for j in range(1, len(walk_order)):
        state = torch.as_tensor(proc.get_state())
        cur_node_ind = graph.index_lookup[walk_order[j].val]                
        update, context = model(state, context, t)
        state = torch.where(state+update>=0., state+update, 0.0)
        state = state/state.sum(axis=-1)
        # dist = Categorical(state)
        # log_prob = dist.log_prob(cur_node_ind)
        t += 1
        W_adj[prev_node_ind, cur_node_ind] = state[cur_node_ind]
        W_adj[cur_node_ind, prev_node_ind] = state[cur_node_ind]
        context[0, cur_node_ind] = 1.
        prev_node_ind = cur_node_ind
        proc.step()
    return W_adj 


def train(args, dags, graph, diffusion_args, norm_props, mol_feats):
    dags_copy = deepcopy(dags)
    dags_copy_train, dags_copy_test, norm_props_train, norm_props_test = train_test_split(dags_copy, norm_props, test_size=0.4, random_state=42)
    all_procs = []

    # data augmentation
    for i in range(len(dags_copy_train)):
        proc = graph.lookup_process(dags_copy_train[i].dag_id)
        procs = [proc]
        if args.augment_dfs:
            for j in range(1, proc.total):
                procs.append(DiffusionProcess(dags_copy_train[i], graph.index_lookup, dfs_seed=j, **graph.diffusion_args))    
        all_procs.append(procs)


    G = graph.graph  
    N = len(G)    

    model = L_grammar(len(G), diffusion_args)
    state = torch.load(os.path.join(args.grammar_folder, 'ckpt.pt'))
    model.load_state_dict(state)

    # Per epoch        
        # For each walk i =: w_i
        # H_i <- EdgeConv(;H, w_i, E) (w_i DAG graph)
        # Optimize E, theta with Loss(MLP(H_i), prop-values)
    
    # init EdgeConv GNN
    predictor = Predictor(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=2)
    if args.predictor_ckpt:
        state = torch.load(args.predictor_ckpt)
        predictor.load_state_dict(state, strict=True)
        for p in predictor.parameters():
            p.requires_grad_(False)
    if args.update_grammar:
        all_params = list(model.parameters()) + list(predictor.parameters())
        if args.predictor_ckpt:
            all_params = list(model.parameters())
    else:
        all_params = list(predictor.parameters())
    if args.opt == 'sgd':
        opt = torch.optim.SGD(all_params, lr=1e-4)         
    elif args.opt == 'adam':
        opt = torch.optim.Adam(all_params)
    else:
        raise
    loss_func = nn.MSELoss()   
    history = []
    train_history = []
    best_loss = float("inf")
    print(args.logs_folder)
    for epoch in range(args.num_epochs):        
        # compute edge control weighted adj matrix via model inference    
        
        graph.reset()
        # random.shuffle(dags_copy)
        train_loss_history = []
        for i in range(len(dags_copy_train)):            
            procs = all_procs[i]
            if i % args.num_accumulation_steps == 1 % args.num_accumulation_steps: opt.zero_grad()            
            for proc in procs:
                W_adj = walk_edge_weight(dags_copy_train[i], graph, model, proc)
                # GNN with edge weight
                node_attr, edge_index, edge_attr = W_to_attr(args, W_adj, mol_feats)            
                X = node_attr
                prop = do_predict(predictor, X, edge_index, edge_attr)                                          
                loss = loss_func(prop, norm_props_train[i])            
                train_loss_history.append(loss.item())     
                loss.backward()
            if args.augment_dfs:
                assert args.num_accumulation_steps == 1
                for p in all_params:
                    if p.requires_grad:
                        p.grad /= len(procs)
            if i % args.num_accumulation_steps == 0:                 
                opt.step()

        loss_history = []
        with torch.no_grad():
            for i in range(len(dags_copy_test)):                        
                W_adj = walk_edge_weight(dags_copy_test[i], graph, model, graph.lookup_process(dags_copy_test[i].dag_id))
                # GNN with edge weight
                node_attr, edge_index, edge_attr = W_to_attr(args, W_adj, mol_feats)            
                X = node_attr
                prop = do_predict(predictor, X, edge_index, edge_attr)                          
                loss = loss_func(prop, norm_props_test[i])            
                loss_history.append(loss.item())                   


        if np.mean(loss_history) < best_loss:
            best_loss = np.mean(loss_history)
            print(f"best_loss epoch {epoch}", best_loss)
            torch.save(predictor.state_dict(), os.path.join(args.logs_folder, f'predictor_ckpt_{best_loss}.pt'))
            if args.update_grammar:
                torch.save(model.state_dict(), os.path.join(args.logs_folder, f'grammar_ckpt_{best_loss}.pt'))
            
        history.append(np.mean(loss_history))
        train_history.append(np.mean(train_loss_history))
        fig_path = os.path.join(args.logs_folder, 'predictor_loss.png')
        fig = plt.Figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(len(history))+1, history, label='test loss')
        ax.plot(np.arange(len(train_history))+1, train_history, label='train loss')
        ax.text(0, min(history), "{}:.3f".format(min(history)))
        ax.axhline(y=min(history), color='red')
        ax.set_title(f"Prediction loss")
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
    for dag in all_dags:
        dag_ids[dag.dag_id] = dag
    for i, l in enumerate(lines):
        if i not in dag_ids: continue
        prop = l.rstrip('\n').split(' ')[-1].split(',')
        prop = list(map(lambda x: float(x) if x != '-' else None, prop))        
        if prop[0] and prop[1]:
            props.append([prop[0],prop[1]])
            dags.append(dag_ids[i])
    
    props = np.array(props)
    mean, std = np.mean(props,axis=0,keepdims=True), np.std(props,axis=0,keepdims=True)    
    with open(os.path.join(logs_folder, 'mean_and_std.txt'), 'w+') as f:
        for i in range(props.shape[-1]):                
            f.write(f"{mean[0,i]},{std[0,i]}\n")
    
    norm_props = torch.FloatTensor((props-mean)/std)  
    return props, norm_props, dags  


def do_predict(predictor, X, edge_index, edge_attr):
    # try modifying X based on edge_attr
    return predictor(X, edge_index, edge_attr)    


def W_to_attr(args, W_adj, mol_feats):
    edge_index = W_adj.nonzero().T
    edge_attr = W_adj.flatten()[W_adj.flatten()>0][:, None]    
    if args.mol_feat == 'W':
        node_attr = W_adj
    else:
        node_attr = torch.as_tensor(mol_feats)
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
        # leaf_node, root_node, e = conn[-1]
        # leaf_node.add_child((root_node, e)) # breaks dag
        # root_node, leaf_node, e = conn[-2]
        # root_node.parent = (leaf_node, e)
        root_node.dag_id = k
        dags.append(root_node)

    config_json = json.loads(json.load(open(os.path.join(args.grammar_folder,'config.json'),'r')))
    diffusion_args = {k[len('diffusion_'):]: v for (k, v) in config_json.items() if 'diffusion' in k}

    graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
    graph = DiffusionGraph(dags, graph, **diffusion_args)     
    G = graph.graph
    N = len(G)    
    mols = load_mols(args.motifs_folder)
    red_grps = annotate_extra(mols, args.extra_label_path)    
    r_lookup = r_member_lookup(mols) 

    if args.mol_feat == 'fp':
        feat_lookup = {}
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
    else:
        raise

    
    setattr(args, 'input_dim', len(mol_feats[0]))

    if args.predictor_file and args.grammar_file:    
        setattr(args, 'logs_folder', os.path.dirname(args.predictor_file))            
        model = L_grammar(N, diffusion_args)
        state = torch.load(args.grammar_file)
        model.load_state_dict(state)
        predictor = Predictor(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=2)
        state = torch.load(os.path.join(args.predictor_file))
        predictor.load_state_dict(state)  
        props, norm_props, dags = preprocess_data(dags, args, os.path.dirname(args.predictor_file))      
    else:    
        predictor_path = os.path.join(args.grammar_folder,f'predictor_{time.time()}')
        os.makedirs(predictor_path, exist_ok=True)
        setattr(args, 'logs_folder', predictor_path)
        with open(os.path.join(predictor_path, 'config.json'), 'w+') as f:
            json.dump(json.dumps(args.__dict__), f)  

        props, norm_props, dags = preprocess_data(dags, args, args.logs_folder)            
        model, predictor = train(args, dags, graph, diffusion_args, norm_props, mol_feats)

    all_nodes = list(G.nodes())  
    predefined_graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)    
    graph.reset()
    trajs = []
    novel = []   
    lines = open(args.walks_file).readlines()  
    walks = set()
    for i, l in enumerate(lines):        
        walk = l.rstrip('\n').split(' ')[0]
        walks.add(walk)
    new_novel = 1
    while len(novel) < args.num_generate_samples and new_novel:
        print(f"add {new_novel} samples, now {len(novel)} novel samples")
        new_novel = 0
        for _ in range(100):
            for n in G.nodes():
                if ':' in n: continue
                traj, good = sample_walk(n, G, graph, model, all_nodes)                
                if len(traj) > 1 and good:
                    name_traj, side_chain = process_good_traj(traj, all_nodes)  
                    assert len(traj) == len(name_traj)
                    try:        
                        root, edge_conn = verify_walk(r_lookup, predefined_graph, name_traj)                           
                        name_traj = '->'.join(name_traj)
                        trajs.append(name_traj)
                        # print(name_traj, "success")
                        if name_traj not in walks:
                            print(name_traj, "novel")
                            walks.add(name_traj)                        
                            proc = DiffusionProcess(root, graph.index_lookup, **diffusion_args)
                            W_adj = walk_edge_weight(root, graph, model, proc)
                            node_attr, edge_index, edge_attr = W_to_attr(args, W_adj, mol_feats)
                            X = node_attr
                            prop = do_predict(predictor, X, edge_index, edge_attr)
                            print("predicted prop", prop)    
                            novel.append((name_traj, root, edge_conn, prop.detach().numpy()))                    
                            new_novel += 1
                    except:
                        pass
            
    orig_preds = []    
    graph.reset()
    loss_history = []
    for i, dag in enumerate(dags):
        W_adj = walk_edge_weight(dag, graph, model, graph.lookup_process(dag.dag_id))
        node_attr, edge_index, edge_attr = W_to_attr(args, W_adj, mol_feats)
        X = node_attr
        prop = do_predict(predictor, X, edge_index, edge_attr)           
        loss_history.append(nn.MSELoss()(prop, norm_props[i]).item())
        prop_npy = prop.detach().numpy()
        orig_preds.append(prop_npy)

    print(np.mean(loss_history))

    print("best novel samples")
    mean = [] ; std = []
    with open(os.path.join(os.path.dirname(args.predictor_file), 'mean_and_std.txt')) as f:
        while True:
            line = f.readline()
            if not line: break
            prop_mean, prop_std = map(float, line.split(','))
            mean.append(prop_mean)
            std.append(prop_std)
            

    out = []
    out_2 = []
    with open(os.path.join(args.logs_folder, 'novel_props.txt'), 'w+') as f:
        f.write("walk,H_2,N_2,H_2/N_2\n")
        for x in novel:
            unnorm_prop = [x[-1][i]*std[i]+mean[i] for i in range(2)]
            out.append(unnorm_prop[0])            
            out_2.append(unnorm_prop[0]/unnorm_prop[1])
            print(f"{x[0]},{','.join(list(map(str,unnorm_prop)))},{unnorm_prop[0]/unnorm_prop[1]}")
            f.write(f"{x[0]},{','.join(list(map(str,unnorm_prop)))},{unnorm_prop[0]/unnorm_prop[1]}\n")
    
    orig_preds = [[orig_pred[i]*std[i]+mean[i] for i in range(2)] for orig_pred in orig_preds]
    out, out_2, orig_preds = np.array(out), np.array(out_2), np.array(orig_preds)
    props[:,1] = props[:,0]/props[:,1]
    orig_preds[:,1] = orig_preds[:,0]/orig_preds[:,1]
    p1, p2 = np.concatenate((out,props[:,0])), np.concatenate((out_2,props[:,1]))    
    pareto_1, not_pareto_1, pareto_2, not_pareto_2 = pareto_or_not(p1, p2, len(out), min_better=False)
    fig = plt.Figure()    
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('(H_2, H_2/N_2) of original vs novel monomers')
    ax.scatter(out[not_pareto_1], out_2[not_pareto_1], c='b', label='predicted values of novel monomers')
    ax.scatter(out[pareto_1], out_2[pareto_1], c='b', marker='v')    
    ax.scatter(props[:,0][not_pareto_2], props[:,1][not_pareto_2], c='g', label='ground-truth values of original monomers')
    ax.scatter(props[:,0][pareto_2], props[:,1][pareto_2], c='g', marker='v')
    ax.scatter(orig_preds[:,0], orig_preds[:,1], c='r', label='predicted values of original monomers')
    ax.set_xlabel('Permeability H2')
    ax.set_ylabel('Selectivity H_2/N_2')
    ax.set_ylim(ymin=0)
    ax.legend()
    ax.grid(True)    
    fig.savefig(os.path.join(args.logs_folder, 'pareto.png'))
    all_walks = {}

    write_conn = lambda conn: [(str(a.id), str(b.id), a.val, b.val, str(e), predefined_graph[a.val][b.val][e]) for (a,b,e) in conn]
    all_walks['old'] = list(data_copy.values())
    novel = sorted(novel, key=lambda x:len(x[2]))
    
    with open(os.path.join(args.logs_folder, 'novel.txt'), 'w+') as f:
        for n in novel:
            f.write(n[0]+'\n')

    all_walks['novel'] = [x[2] for x in novel]  
    # pickle.dump(novel, open(os.path.join(args.logs_folder, 'novel.pkl', 'wb+')))
    all_walks['old'] = [write_conn(x[-1]) for x in all_walks['old']]
    all_walks['novel'] = [write_conn(x) for x in all_walks['novel']]
    print("novel", novel)
    json.dump(all_walks, open(os.path.join(args.logs_folder, 'all_dags.json'), 'w+'))

                
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motifs_folder')    
    parser.add_argument('--extra_label_path')    
    parser.add_argument('--grammar_folder')
    parser.add_argument('--grammar_file', help='if provided, sample new')
    parser.add_argument('--predictor_file', help='if provided, sample new')
    parser.add_argument('--predictor_ckpt')
    parser.add_argument('--update_grammar', action='store_true')

    # data params
    parser.add_argument('--predefined_graph_file')    
    parser.add_argument('--dags_file')
    parser.add_argument('--walks_file')    
    parser.add_argument('--augment_dfs', action='store_true')

    # training params
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_accumulation_steps', type=int, default=1)
    parser.add_argument('--opt', default='adam')

    # model params
    parser.add_argument('--mol_feat', type=str, choices=['fp', 'one_hot', 'ones', 'W'])
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=5)

    # sampling params
    parser.add_argument('--num_generate_samples', type=int, default=100)

    args = parser.parse_args()    
     
    main(args)
