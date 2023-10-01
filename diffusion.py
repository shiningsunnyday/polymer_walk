import argparse
import pickle
import numpy as np
from utils import *
import networkx as nx
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import json
from copy import deepcopy
from torch_geometric.nn.conv import GINEConv


class DiffusionProcess:
    """
    We non-lazily diffuse a particle across a DAG.
    Given a dag (monomer), we model the particle's location probabilistically.
    1. A leaf node, L, is connected to the root node, R, meaning the main chain becomes a cycle. 
    This breaks the DAG but is used for non-lazy diffusion.
    2. The particle on the main chain must do one of:
    - stay on the main chain and go to the next node
    - descend an unvisited side chain
    3. The particle on the side chain, must:
    - descend the side chain if not visited the leaf yet
    - ascend where it came from if visited the leaf already
    """
    def __init__(self, dag, lookup, side_chains=False, split=False, **diffusion_args):
        self.lookup = lookup
        self.dag = dag
        self.side_chains = side_chains
        self.main_chain = []
        self.split = split
        self.reset()
        if not self.split:
            res = []
            self.dfs_walk(dag, res)
            self.dfs_order = res
            self.num_nodes = len(res)


    @staticmethod
    def compute_main_chain(dag):
        chain = [dag]
        while len(chain) == 1 or chain[-1].id:
            for child, _ in chain[-1].children:
                if child.side_chain: continue
                chain.append(child)
                break
        return chain


    @staticmethod
    def dfs_walk(node, res):
        res.append(node)
        childs = sorted(node.children, key=lambda x: (not x[0].side_chain, x[0].id))
        for c in childs:
            side_chain = c[0].side_chain
            ind = len(res)-1
            if c[0].id:
                DiffusionProcess.dfs_walk(c[0], res)
                if side_chain:
                    for i in range(len(res)-2, ind-1, -1):
                        res.append(res[i])
                    # print([a.val for a in res[:ind+1]], [a.val for a in res], "before after")


    
    def reset(self):
        self.t = 0
        self.state = np.array([0.0 for _ in self.lookup])
        self.state[self.lookup[self.dag.val]] = 1.0
        self.frontier = {self.dag: 1}


    def step(self):
        new_frontier = defaultdict(float)
        if self.split:
            for cur, p in self.frontier.items():
                if cur.side_chain:
                    if not self.side_chains: continue
                    if cur.children:
                        breakpoint()
                        pass
                    else:
                        new_frontier[cur.parent[0]] += p
                else:     
                    for a in cur.children:
                        new_frontier[a[0]] += p/len(cur.children)
        else:
            new_frontier[self.dfs_order[(self.t+1)%self.num_nodes]] = 1.0
            
                
        new_state = np.zeros(len(self.state))
        for k, v in new_frontier.items():
            new_state[self.lookup[k.val]] = v
        if new_state.sum() - 1. > 0.01:
            breakpoint()
        self.state = new_state
        self.frontier = new_frontier
        self.t += 1

    
    def get_state(self):
        return self.state


class DiffusionGraph:
    """
    This abstracts n simultaneous diffusion processes as one process on a single graph.
    It slightly modifies the predefined graph, since some monomers use the same group k times (k>1).
    In that case, the modified graph must have k replicates of the same group.
    """
    def __init__(self, dags, graph, **diffusion_args):
        self.dags = dags   
        self.diffusion_args = diffusion_args
        self.t = 0        
        self.processes = []
        self.index_lookup = self.modify_graph(dags, graph)
        for dag in dags:
            self.processes.append(DiffusionProcess(dag, self.index_lookup, **diffusion_args))
        self.graph = graph


    def modify_graph(self, dags, graph):
        max_value_count = {}
        for i, dag in enumerate(dags):
            value_counts = {}
            self.value_count(dag, value_counts)
            for k, v in value_counts.items():
                max_value_count[k] = max(max_value_count.get(k, 0), v)
        
        for k, count in max_value_count.items():
            if count == 1: continue
            for i in range(count):
                graph.add_node(k+f':{i+1}')
                for dest, e_data in list(graph[k].items()):
                    for key, v in e_data.items():
                        graph.add_edge(k+f':{i+1}', dest, **v)
                    for key, v in graph[dest][k].items():
                        graph.add_edge(dest, k+f':{i+1}', **v)
            
            if k in graph[k]: # no self-loops anymore
                k_ = [k]+[k+f':{i+1}' for i in range(count)]
                for a, b in product(k_, k_):
                    if b in graph[a]: graph.remove_edge(a,b)
        
        return dict(zip(list(graph.nodes()), range(len(graph.nodes()))))            

    @staticmethod
    def value_count(node, counts):
        counts[node.val] = counts.get(node.val,0)+1
        if counts[node.val]>1:
            node.val += f':{counts[node.val]-1}'
        for c in node.children:
            try:
                if c[0].id == 0: continue
            except:
                breakpoint()
            DiffusionGraph.value_count(c[0], counts)

    def reset(self):
        self.t = 0
        for p in self.processes:
            p.reset()

    def step(self):
        for p in self.processes:
            p.step()
        self.t += 1
    
    def get_state(self, return_all=False):
        all_probs = [p.get_state() for (i, p) in enumerate(self.processes)]
        if return_all:
            return np.array(all_probs)
        probs = np.array(all_probs).sum(axis=0, keepdims=True)
        assert probs.shape[0] == 1
        return probs/probs.sum()
    

class L_grammar(nn.Module):
    def __init__(self, N, diff_args):
        super().__init__()
        self.diff_args = diff_args
        if diff_args['e_init']:
            E = torch.as_tensor(nx.adjacency_matrix(G).toarray(), dtype=torch.float64)
        else:
            E = torch.zeros((N, N), dtype=torch.float64)    
        self.E = E    
        self.scale = nn.Parameter(torch.ones((1,), dtype=torch.float64))
        self.W = nn.Parameter(E)
        self.A = torch.as_tensor(E.clone().detach(), dtype=torch.float64)
        self.context_layer = nn.Linear(N, N*N, dtype=torch.float64)
        nn.init.zeros_(self.context_layer.weight)
        nn.init.zeros_(self.context_layer.bias)
                 

    def forward(self, X, context, t):
        if self.diff_args['combine_walks']:
            L = torch.diag((self.W*self.A).sum(axis=0))-self.W # (N, N)
        else:
            if self.diff_args['context_L']:
                adjust = self.context_layer(context).reshape((-1,)+self.W.shape)
                W_new = adjust + self.W # (M, N, N)
            else:
                W_new = self.W
            L = torch.diag_embed(torch.matmul(W_new,self.A[None]).sum(axis=-2))-W_new # (M, N, N)
        
        context = context*t/(t+1) + X/(t+1)              
        L_T = self.scale * torch.transpose(L, -1,-2)
        update = torch.matmul(X[...,None,:],L_T).squeeze()
        return update, context
    

class Predictor(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(), 
                            nn.Linear(hidden_dim, hidden_dim))
        self.out_mlp = nn.Linear(hidden_dim, 1)
        self.gnn = GINEConv(mlp, edge_dim=1)
        self.num_layers = num_layers

    

    def forward(self, X, edge_index, edge_attr):
        for _ in range(self.num_layers):
            X = self.gnn(X, edge_index, edge_attr)
        prop = self.out_mlp(X.sum(axis=0))          
        return prop    

def diffuse(graph, log_folder, **diff_args):
    G = graph.graph
    print(f"state at 0: {graph.get_state()}")    
    N, M = len(G), 1 if diff_args['combine_walks'] else len(graph.processes)
    model = L_grammar(N, diff_args)
    # if diff_args['e_init']:
    #     E = torch.as_tensor(nx.adjacency_matrix(G).toarray(), dtype=torch.float64)
    # else:
    #     E = torch.zeros((N, N), dtype=torch.float64)
    # scale = nn.Parameter(torch.ones((1,), dtype=torch.float64))
    # W = nn.Parameter(E)
    # A = torch.as_tensor(E.clone().detach(), dtype=torch.float64)
    # context_layer = nn.Linear(N, N*N, dtype=torch.float64)
    # nn.init.zeros_(context_layer.weight)
    # nn.init.zeros_(context_layer.bias)
    # loss_func = nn.MSELoss()   
    # parameters = [W, scale]+list(context_layer.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=diff_args['alpha'])
    history = []
    T = 10
    for i in range(diff_args['num_epochs']):
        graph.reset()
        context = torch.zeros((M, N), dtype=torch.float64)
        loss_func = nn.MSELoss()    
        for t in range(T):
            t_losses = []
            opt.zero_grad()     
            X = torch.as_tensor(graph.get_state(not diff_args['combine_walks'])) # (M, N)
            graph.step()
            Y = torch.as_tensor(graph.get_state(not diff_args['combine_walks'])) # (M, N)                      
            update, context = model(X, context, t)
            loss = loss_func(X+update, Y) # (1,N)+(1,N)(N,N) or (M,N)+(M,1,N)(M,N,N)        
            t_losses.append(loss.item())
            loss.backward()
            opt.step()


        print(f"epoch {i} loss: {np.mean(t_losses)}")
        history.append(np.mean(t_losses))
        
        if i % 1000: continue
        fig = plt.Figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(history)
        ax.text(0, min(history), "{}".format(min(history)))
        ax.axhline(y=min(history), color='red')
        ax.set_title(f"Loss over {diff_args['num_epochs']} epochs, {T} steps each")
        ax.set_ylabel(f"MSE Loss of X^t")
        ax.set_xlabel('(Epoch, t)')
        plot_file = os.path.join(log_folder, 'L_loss.png')
        fig.savefig(plot_file)
        print(plot_file)
    torch.save(model.state_dict(), os.path.join(log_folder, 'ckpt.pt'))
    return model


def side_chain_grammar(index_lookup, log_folder):
    X = [] ; y = []
    history = []
    drop_colon = lambda x: x.split(':')[0]
    for dag in dags:
        chain = DiffusionProcess.compute_main_chain(dag)
        hot = torch.LongTensor([index_lookup[drop_colon(n.val)] for n in chain])
        walk = F.one_hot(hot, num_classes=num_nodes).sum(axis=0, keepdims=True)
        layer = nn.Linear(2*num_nodes, 2)
        for n in chain:
            cur = torch.LongTensor([index_lookup[drop_colon(n.val)]])
            walk_cur = torch.cat((F.normalize(walk+0.0), F.one_hot(cur, num_classes=num_nodes)+0.0), dim=-1)
            X.append(walk_cur)
            side_chain = 0
            for c in n.children:
                if c[0].side_chain:
                    side_chain = 1
            y.append(side_chain)

    X, y = torch.cat(X, dim=0), torch.tensor(y)
    print(X.shape, y.shape)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    loss_min = float("inf")
    min_i = 0
    i = 0
    while True:
        opt.zero_grad()
        loss = loss_func(F.sigmoid(layer(X)), y)
        loss.backward()
        opt.step()
        history.append(loss.item())
        if loss.item() < loss_min:
            loss_min = loss.item()
            min_i = i
        
        if i - min_i > 10:
            print(f"converged at {min_i}")
            break
            
        i += 1

    pickle.dump(layer, open(os.path.join(log_folder, 'side_chain_grammar.pkl'), 'wb+'))
    print(os.path.join(log_folder, 'side_chain_grammar.pkl'))
    fig = plt.Figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(history)    
    ax.set_title(f"Training side chain grammar")
    ax.set_ylabel(f"Binary cross entropy loss")
    ax.set_xlabel('Epoch')
    plot_file = os.path.join(log_folder, 'side_chain_grammar_loss.png')
    fig.savefig(plot_file)
    print(plot_file)    
    return layer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dags_file')
    parser.add_argument('--data_file')
    parser.add_argument('--motifs_folder')
    parser.add_argument('--extra_label_path')    
    parser.add_argument('--predefined_graph_file')
    parser.add_argument('--log_folder')
    # diffusion args
    parser.add_argument('--side_chains', dest='diffusion_side_chains', action='store_true')
    parser.add_argument('--split', dest='diffusion_split', action='store_true')
    parser.add_argument('--combine_walks', dest='diffusion_combine_walks', action='store_true')
    parser.add_argument('--e_init', dest='diffusion_e_init', action='store_true')
    parser.add_argument('--context_L', dest='diffusion_context_L', action='store_true')
    parser.add_argument('--alpha', dest='diffusion_alpha', default=1e-4, type=float)
    parser.add_argument('--num_epochs', dest='diffusion_num_epochs', default=500, type=int)

    args = parser.parse_args()
    lines = open(args.data_file).readlines()   
    walks = set()
    for i, l in enumerate(lines):        
        walk = l.rstrip('\n').split(' ')[0]
        walks.add(walk)
    print(walks)
    diffusion_args = {k[len('diffusion_'):]: v for (k, v) in args.__dict__.items() if 'diffusion' in k}

    graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
    predefined_graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
    mols = load_mols(args.motifs_folder)
    red_grps = annotate_extra(mols, args.extra_label_path)  
    r_lookup = r_member_lookup(mols)

    num_nodes = len(graph.nodes())
    index_lookup = dict(zip(graph.nodes(), range(num_nodes)))
    data = pickle.load(open(args.dags_file, 'rb'))
    data_copy = deepcopy(data)
    dags = []        
    for k, v in data.items():
        grps, root_node, conn = v        
        leaf_node, root_node, e = conn[-1]
        leaf_node.add_child((root_node, e)) # breaks dag
        root_node, leaf_node, e = conn[-2]
        root_node.parent = (leaf_node, e)
        root_node.dag_id = k
        dags.append(root_node)
        

    graph = DiffusionGraph(dags, graph, **diffusion_args) 
    G = graph.graph
    all_nodes = list(G.nodes())
    if args.log_folder:
        model = L_grammar(len(graph.graph), diffusion_args)
        state = torch.load(os.path.join(args.log_folder, 'ckpt.pt'))
        model.load_state_dict(state)
        E = model.E
        E_dic = defaultdict(dict)
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                a, b = all_nodes[i], all_nodes[j]
                E_dic[a][b] = E[i][j].item()
        json.dump(E_dic, open(os.path.join(args.log_folder, 'E.json'), 'w+'))
    else:        
        log_dir = os.path.join('logs/', f'logs-{time.time()}/')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'config.json'), 'w+') as f:
            json.dump(json.dumps(args.__dict__), f)        
        model = diffuse(graph, log_dir, **diffusion_args)
    
    graph.reset()
    trajs = []
    novel = []

    if args.diffusion_side_chains:
        layer = side_chain_grammar(index_lookup, args.log_folder)

    N = len(G)
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
            side_chains = {} # index in traj to side chain
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
                        novel.append((name_traj, root, edge_conn))
                except:
                    print(name_traj, "failed")
                

    all_walks = {}
    write_conn = lambda conn: [(str(a.id), str(b.id), a.val, b.val, str(e), predefined_graph[a.val][b.val][e]) for (a,b,e) in conn]
    all_walks['old'] = list(data_copy.values())
    novel = sorted(novel, key=lambda x:len(x[2]))
    all_walks['novel'] = novel
    with open(os.path.join(args.log_folder, 'novel.txt'), 'w+') as f:
        for n in novel:
            f.write(n[0]+'\n')

    # pickle.dump(novel, open(os.path.join(args.logs_folder, 'novel.pkl', 'wb+')))

    all_walks['old'] = [write_conn(x[-1]) for x in all_walks['old']]
    all_walks['novel'] = [write_conn(x[-1]) for x in all_walks['novel']]
    print("novel", novel)
    json.dump(all_walks, open(os.path.join(args.log_folder, 'all_dags.json'), 'w+'))
    