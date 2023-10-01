import argparse
import pickle
import networkx as nx
from utils.graph import *
from diffusion import L_grammar, Predictor, DiffusionGraph, DiffusionProcess
import json
import torch
import time
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def walk_edge_weight(dag, graph, model, proc):
    N = len(graph.graph)
    walk_order = []
    DiffusionProcess.dfs_walk(dag, walk_order)    
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


def train(args, graph, all_dags, diffusion_args):
    lines = open(args.walks_file).readlines()
    props = []
    dag_ids = [None for i in range(len(lines))]
    dags = []    
    for dag in all_dags:
        dag_ids[dag.dag_id] = 1
    for i, (dag_id, l) in enumerate(zip(dag_ids, lines)):
        if dag_id == None: continue
        prop = l.rstrip('\n').split(' ')[-1].split(',')
        prop = list(map(lambda x: float(x) if x != '-' else None, prop))        
        if prop[0]:
            props.append(prop[0])
            dags.append(dag)
    
    mean, std = np.mean(props), np.std(props)
    with open(os.path.join(args.logs_folder, 'mean_and_std.txt'), 'w+') as f:
        f.write(f"{mean},{std}\n")
    norm_props = torch.FloatTensor([[(p-mean)/std] for p in props])

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
    hidden_dim = 16
    num_layers = 5
    predictor = Predictor(hidden_dim, num_layers)
    if args.update_grammar:
        all_params = list(model.parameters()) + list(predictor.parameters())
    else:
        all_params = list(predictor.parameters())
    opt = torch.optim.Adam(all_params)
    loss_func = nn.MSELoss()   
    history = []
    best_loss = float("inf")
    for _ in range(args.num_epochs):        
        # compute edge control weighted adj matrix via model inference    
        graph.reset()
        for i in range(len(dags)):            
            loss_history = []
            W_adj = walk_edge_weight(dags[i], graph, model, graph.processes[i])
            # GNN with edge weight
            edge_index = W_adj.nonzero().T
            edge_attr = W_adj.flatten()[W_adj.flatten()>0][:, None]
            node_attr = torch.ones((W_adj.shape[0], hidden_dim), dtype=torch.float32)
            X = node_attr
            try:
                prop = predictor(X, edge_index, edge_attr)
            except:
                breakpoint()
            
            opt.zero_grad()        
            loss = loss_func(prop, norm_props[i])
            loss_history.append(loss.item())
            loss.backward()
            opt.step()

        if np.mean(loss_history) < best_loss:
            best_loss = np.mean(loss_history)
            torch.save(predictor.state_dict(), os.path.join(args.logs_folder, f'predictor_ckpt_{best_loss}.pt'))
            if args.update_grammar:
                torch.save(model.state_dict(), os.path.join(args.logs_folder, f'grammar_ckpt_{best_loss}.pt'))
            
        history.append(np.mean(loss_history))
        fig_path = os.path.join(args.logs_folder, 'predictor_loss.png')
        fig = plt.Figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(len(history))+1, history)
        ax.text(0, min(history), "{}".format(min(history)))
        ax.axhline(y=min(history), color='red')
        ax.set_title(f"Prediction loss")
        ax.set_ylabel(f"MSE Loss")
        ax.set_xlabel('Epoch')
        fig.savefig(fig_path)  

    return model, predictor      



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
    hidden_dim = 16
    if args.predictor_file and args.grammar_file:        
        model = L_grammar(N, diffusion_args)
        state = torch.load(args.grammar_file)
        model.load_state_dict(state)
        predictor = Predictor(hidden_dim=16, num_layers=5)
        state = torch.load(os.path.join(args.predictor_file))
        predictor.load_state_dict(state)        
    else:    
        predictor_path = os.path.join(args.grammar_folder,f'predictor_{time.time()}')
        os.makedirs(predictor_path, exist_ok=True)
        setattr(args, 'logs_folder', predictor_path)
        with open(os.path.join(predictor_path, 'config.json'), 'w+') as f:
            json.dump(json.dumps(args.__dict__), f)            
        model, predictor = train(args, graph, dags, diffusion_args)

    all_nodes = list(G.nodes())  
    mols = load_mols(args.motifs_folder)
    red_grps = annotate_extra(mols, args.extra_label_path)    
    r_lookup = r_member_lookup(mols) 
    predefined_graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)    
    graph.reset()
    trajs = []
    novel = []   
    lines = open(args.walks_file).readlines()  
    walks = set()
    for i, l in enumerate(lines):        
        walk = l.rstrip('\n').split(' ')[0]
        walks.add(walk)     
    for _ in range(1000):
        for n in G.nodes():
            if ':' in n: continue
            context = torch.zeros((1, N), dtype=torch.float64)
            start = graph.index_lookup[n]
            state = torch.zeros((1, len(G)), dtype=torch.float64)
            state[0, graph.index_lookup[n]] = 1.
            traj = [str(start)]
            t = 0
            after = -1
            good = False
            while True:
                update, context = model(state, context, t)
                if not (state>=0).all():
                    breakpoint()
                state = torch.where(state+update>=0., state+update, 0.0)
                state = state/state.sum(axis=-1)
                t += 1
                state_numpy = state.detach().flatten().numpy()
                after = np.random.choice(len(state_numpy), p=state_numpy)    
                if ':' in all_nodes[after]:
                    ind = int(all_nodes[after].split(':')[-1])
                    bad_ind = False
                    grp = all_nodes[after].split(':')[0]
                    prev_indices = [all_nodes[extract(x)] for x in traj if grp in all_nodes[extract(x)]]
                    for prev_ind in prev_indices:
                        if ':' in prev_ind and int(prev_ind.split(':')[-1]) > ind:
                            bad_ind = True # P3:4 seen but we get 'P3:3'
                    for i in range(ind-1, -1, -1):
                        prev_ind_str = grp+(':'+str(i) if i else '')
                        if prev_ind_str not in prev_indices:
                            bad_ind = True # we get P3:3 but no P3:2 seen
                    
                    if bad_ind: break
                
                state = torch.zeros(len(G), dtype=torch.float64)
                state[after] = 1.
                if extract(traj[-1]) == after:
                    traj.append(str(after))
                    break
                if after == start:
                    traj.append(str(after))
                    good = True
                    break
                # after indicates side chain, e.g. A, B, A good but A, B, C, A bad
                def extract_sides(x):
                    occur = []
                    occur.append(x.split('[')[0])
                    for a in x.split('[')[1][:-1].split(','):
                        occur.append(a.split('->')[-1])
                    return occur


                occur = []
                for x in traj:
                    if '[' in occur:
                        occur += extract_sides(x)
                    else:
                        occur.append(x)                
                occur = np.array([str(after) in x for x in occur])
                if occur.sum():
                    if len(occur) == 1 or occur.sum() != 1: break
                    if str(after) != traj[-2].split('[')[0]: break
  
                    if '[' in traj[-2]:
                        traj[-2] = traj[-2][:-1]+',->'+str(traj[-1])+']'
                    else:
                        traj[-2] = f"{traj[-2]}[->{traj[-1]}]"
                    traj.pop(-1)
                else:
                    traj.append(str(after))
            
            if len(traj) > 1 and good:
                drop_colon = lambda x: x.split(':')[0]
                name_traj = []
                side_chain = False                
                for x in traj:
                    if '[' in x:
                        side_chain = True
                        side = x.split('[')[1][:-1]
                        new_side = []
                        for y in side.split(','):
                            name = all_nodes[int(y[len('->'):])]
                            new_side.append('->' + name.split(':')[0])
                        side = ','.join(new_side)                        
                        c = all_nodes[int(drop_colon(x.split('[')[0]))]+'['+side+']'
                    else:
                        c = all_nodes[int(x)]
                        if ':' in c:
                            c = drop_colon(c)
                    
                    name_traj.append(c)
                
                assert len(traj) == len(name_traj)

                try:        
                    root, edge_conn = verify_walk(r_lookup, predefined_graph, name_traj)                           
                    name_traj = '->'.join(name_traj)
                    trajs.append(name_traj)
                    print(name_traj, "success")
                    if name_traj not in walks:
                        print(name_traj, "novel")
                        walks.add(name_traj)                        
                        proc = DiffusionProcess(root, graph.index_lookup, **diffusion_args)
                        W_adj = walk_edge_weight(root, graph, model, proc)
                        edge_index = W_adj.nonzero().T
                        edge_attr = W_adj.flatten()[W_adj.flatten()>0][:, None]
                        node_attr = torch.ones((W_adj.shape[0], hidden_dim), dtype=torch.float32)
                        X = node_attr
                        prop = predictor(X, edge_index, edge_attr)
                        print("predicted prop", prop)    

                        novel.append((name_traj, root, edge_conn, prop.item()))                    
                except:
                    print(name_traj, "failed")
        
    print("best novel samples")
    with open(os.path.join(os.path.dirname(args.predictor_file), 'mean_and_std.txt')) as f:
        mean, std = map(float, f.readline().split(','))

    with open(os.path.join(os.path.dirname(args.predictor_file), 'novel_props.txt'), 'w+') as f:
        f.write("walk,prop_val\n")
        for x in sorted(novel, key=lambda x: x[-1]):
            unnorm_prop = x[-1]*std+mean
            print(f"{x[0]},{unnorm_prop}")            
            f.write(f"{x[0]},{unnorm_prop}\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motifs_folder')    
    parser.add_argument('--extra_label_path')    
    parser.add_argument('--grammar_folder')
    parser.add_argument('--dags_file')
    parser.add_argument('--walks_file')
    parser.add_argument('--grammar_file', help='if provided, sample new')
    parser.add_argument('--predictor_file', help='if provided, sample new')
    parser.add_argument('--predefined_graph_file')
    parser.add_argument('--update_grammar', action='store_true')
    parser.add_argument('--num_epochs', type=int)
    args = parser.parse_args()    
     
    main(args)
