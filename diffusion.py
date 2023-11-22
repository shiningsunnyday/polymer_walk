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
from math import factorial
from torch_geometric.nn.conv import GINEConv, GINConv, GATConv, GCNConv


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
    def __init__(self, dag, lookup, side_chains=False, split=False, dfs_seed=0, **diffusion_args):
        self.lookup = lookup
        self.dag = dag
        self.side_chains = side_chains
        self.main_chain = DiffusionProcess.compute_main_chain(dag)
        self.child_nums = [len(self.side_childs(a)) for a in self.main_chain[:-1]]
        self.total = np.prod([factorial(x) for x in self.child_nums])        
        self.split = split
        self.reset()
        if not self.split:
            self.dfs_dir = dfs_seed >= 0
            dfs_seed = abs(dfs_seed)
            self.dfs_seed = dfs_seed % self.total # 0 to X-1, where X := prod_(node in main chain) num_childs(node)            
            res = self.compute_dfs(dag)
            new_res = self.augment_walk_order(res, dfs_seed)
            self.dfs_order = new_res
            self.num_nodes = len(res)



    def augment_walk_order(self, res, dfs_seed):
        # dfs_seed // self.total is which node to start in main chain
        indices = []
        i = 0
        for j in range(len(res)):
            if res[j] == self.main_chain[i]:
                indices.append(j)
                i += 1
        new_res = []
        start_node = dfs_seed//self.total
        for step in range(len(indices)):
            if self.dfs_dir:
                start = (start_node+step)%len(indices)
                end = (start_node+step+1)%len(indices)
            else:
                start = (start_node-step-1+len(indices))%len(indices)
                end = (start_node-step+len(indices))%len(indices)            
            if indices[end] < indices[start]:
                new_res += (res[indices[start]:] + res[:indices[end]])
            else:
                new_res += res[indices[start]: indices[end]]

        return new_res


    def compute_dfs(self, dag):        
        res = []        
        dfs_seed = self.dfs_seed
        perm_map = {}
        for i in range(len(self.child_nums)-1, -1, -1):
            perm_idx = dfs_seed % factorial(self.child_nums[i])
            perm = list(permutations(self.side_childs(self.main_chain[i])))[perm_idx]
            perm_map[self.main_chain[i].id] = perm
            dfs_seed //= factorial(self.child_nums[i])
        assert dfs_seed == 0
        self.dfs_walk(dag, res, perm_map)
        return res


    @staticmethod
    def side_childs(a):
        return [x for x in a.children if x[0].side_chain]        
    

    @staticmethod
    def compute_main_chain(dag):
        chain = [dag]
        i = 0
        while len(chain) == 1 or chain[-1].id:
            i += 1
            main_chain_child = False
            for c in chain[-1].children:
                if not c[0].side_chain:
                    main_chain_child = True
            if main_chain_child: # exists main chain child
                for child, _ in chain[-1].children:
                    if child.side_chain: continue
                    chain.append(child)
                    break
            else:
                chain.append(dag)
        return chain


    @staticmethod
    def dfs_walk(node, res, perm_map=None):
        """
        perm_map: dict(node: permutation of child indices)
        """
        res.append(node)
        childs = sorted(node.children, key=lambda x: (not x[0].side_chain, x[0].id)) # side chains first
        if perm_map:
            try:
                ind = [c[0].side_chain for c in childs].index(False)
            except:
                ind = len(childs)        
            if not node.side_chain: # reorder the children
                childs[:ind] = perm_map[node.id]
        for c in childs:
            side_chain = c[0].side_chain
            ind = len(res)-1
            if c[0].id:
                DiffusionProcess.dfs_walk(c[0], res, perm_map)
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
            if dag.id:
                breakpoint()
            self.processes.append(DiffusionProcess(dag, self.index_lookup, **diffusion_args))
        self.graph = graph


    def lookup_process(self, dag_id):
        for dag, proc in zip(self.dags, self.processes):
            if dag.dag_id == dag_id:
                return proc
        raise


    def modify_graph(self, dags, graph):
        max_value_count = {}
        for i, dag in enumerate(dags):
            value_counts = {}
            if dag.id:
                breakpoint()
            self.value_count(dag, value_counts)
            if dag.id:
                breakpoint()
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
            E = torch.as_tensor(diff_args['init_e'], dtype=torch.float64)
        else:
            E = torch.zeros((N, N), dtype=torch.float64)    
        self.E = nn.Parameter(E)
        self.scale = nn.Parameter(torch.ones((1,), dtype=torch.float64))
        self.A = torch.as_tensor(E.clone().detach(), dtype=torch.float64)
        self.context_layer = nn.Linear(N, N*N, dtype=torch.float64)
        nn.init.zeros_(self.context_layer.weight)
        nn.init.zeros_(self.context_layer.bias)
                 

    def forward(self, X, context, t):
        if self.diff_args['combine_walks']:
            L = torch.diag((self.E*self.A).sum(axis=0))-self.E # (N, N)
        else:
            if self.diff_args['context_L']:
                adjust = self.context_layer(context).reshape((-1,)+self.E.shape)
                W_new = adjust + self.E # (M, N, N)
            else:
                W_new = self.E
            L = torch.diag_embed(torch.matmul(W_new,self.A[None]).sum(axis=-2))-W_new # (M, N, N)
        
        context = context*t/(t+1) + X/(t+1)              
        L_T = self.scale * torch.transpose(L, -1,-2)
        update = torch.matmul(X[...,None,:],L_T).squeeze()
        return update, context
    

class Predictor(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=16, num_layers=5, num_heads=2, gnn='gin', act='relu'):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gnn = gnn
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        else:
            raise
        # assert input_dim == hidden_dim          
        if gnn == 'gin':
            for i in range(1, num_layers+1):
                if i > 1: input_dim = hidden_dim     
                lin_i_1 = nn.Linear(input_dim, hidden_dim)
                lin_i_2 = nn.Linear(hidden_dim, hidden_dim)
                # nn.init.zeros_(lin_i_1.weight)
                # nn.init.zeros_(lin_i_2.weight)
                # nn.init.zeros_(lin_i_1.bias)
                # nn.init.zeros_(lin_i_2.bias)            
                
                # setattr(self, f"gnn_{i}", GATConv(in_channels=-1, out_channels=hidden_dim, edge_dim=1))
                mlp = nn.Sequential(lin_i_1, act, lin_i_2)
                setattr(self, f"gnn_{i}", GINConv(mlp, edge_dim=1))
        elif gnn == 'gat':
            setattr(self, f"gnn_conv", GATConv(input_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=hidden_dim))
        elif gnn == 'gcn':
            setattr(self, f"gnn_conv", GCNConv(input_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=hidden_dim))
        else:
            raise NotImplementedError            

        for i in range(1, num_heads+1):
            lin_out_1 = nn.Linear(hidden_dim, hidden_dim)
            lin_out_2 = nn.Linear(hidden_dim, 1)
            mlp_out = nn.Sequential(lin_out_1, act, lin_out_2)
            # nn.init.zeros_(lin_out.weight)
            # nn.init.zeros_(lin_out.bias)
            setattr(self, f"out_mlp_{i}", mlp_out)            
        
        

    def forward(self, X, edge_index, edge_weights):     
        if self.gnn == 'gin':
            for i in range(1, self.num_layers+1):
                X = getattr(self, f"gnn_{i}")(X, edge_index)
        elif self.gnn == 'gat':
            X = getattr(self, f"gnn_conv")(X, edge_index, edge_weights)
        elif self.gnn == 'gcn':
            X = getattr(self, f"gnn_conv")(X, edge_index, edge_weights)
        else:
            raise
        props = [getattr(self, f"out_mlp_{i}")(X.sum(axis=0)) for i in range(1,self.num_heads+1)]
        return torch.cat(props, dim=-1)
    

def state_to_probs(state):
    state = torch.where(state>=0., state, 0.0)
    return state/state.sum(axis=-1)
    state = state - state.min(-1, True).values
    return state/state.sum(axis=-1)



def diffuse(graph, log_folder, **diff_args):
    G = graph.graph
    print(f"state at 0: {graph.get_state()}")    
    N, M = len(G), 1 if diff_args['combine_walks'] else len(graph.processes)
    if diff_args['e_init']:
        diff_args['init_e'] = nx.adjacency_matrix(G).toarray()
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
    T = 20
    best_loss = float("inf")
    for i in range(diff_args['num_epochs']):
        graph.reset()
        context = torch.zeros((M, N), dtype=torch.float64)
        loss_func = nn.MSELoss()    
        t_losses = []
        for t in range(T):            
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

        if np.mean(t_losses) < best_loss:
            print(f"E mean: {model.E.mean()}, std: {model.E.std()}")
            best_loss = np.mean(t_losses)
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
    opt = torch.optim.Adam(layer.parameters(), lr=1e-4)
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


def process_good_traj(traj, all_nodes):
    """
    take a numbered traj, like ['61', '90', '50[->12->37,->48]', '90']
    turn it into ['L3','S32','S20[->P14->P39,->S18]','S32']
    simple string parsing algo is enough
    """
    name_traj = []
    for x in traj:
        i = 0
        y = ""
        while i < len(x):
            if x[i].isdigit():
                j = i+1
                while j < len(x):
                    if not x[j].isdigit():
                        break
                    j += 1
                y += all_nodes[int(x[i:j])]
                i = j
            else:
                y += x[i]
                i += 1
        name_traj.append(y)
    return name_traj


# after indicates side chain, e.g. A, B, A good but A, B, C, A bad
def extract_sides(x):
    # L3[->P28,->S20] to L3, P28, S20
    occur = []
    occur.append(x.split('[')[0])
    for a in x.split('[')[1][:-1].split(','):
        occur.append(a.split('->')[-1])
    return occur


def check_colon_order(all_nodes, traj, after):
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
    return bad_ind



def get_repr(state):
    start_inds, end_inds = state.nonzero(as_tuple=True)
    state_repr = []
    
    for a, b in zip(start_inds, end_inds):
        state_repr.append([a.item(),b.item(),round(state[a, b].item(),2)])
    return state_repr



def append_traj(traj, after):
    # convert traj=['P21[->L3,->S20]', 'P20'] into ['P21', 'P3', 'S20', 'P20']
    occur = []
    for x in traj:
        if '[' in x:
            occur += extract_sides(x)
        else:
            occur.append(x)

    occur = np.array([str(after) in x for x in occur])
    
    # convert L3, S21, L3 into L3[->S21]
    # convert L3[->P28,->S20], S21, L3 into L3[->P28,->S20,->S21]
    if occur.sum():
        if len(occur) == 1 or occur.sum() != 1: return []
        if str(after) != traj[-2].split('[')[0]: return []
        print("before", traj, after)
        if '[' in traj[-2]:
            if '[' in traj[-1]:
                # example: ['90', '50[->8]', '4[->25]'] 50
                # linearize traj[-1] first
                side = ''.join([f"->{traj[-1][:traj[-1].find('[')]}"] + traj[-1][traj[-1].find('[')+1:-1].split(','))
                # ->4->25
                assert ']' == traj[-2][-1]
                traj[-2]= f"{traj[-2][:-1]},{side}]"
            else:
                traj[-2] = traj[-2][:-1]+',->'+str(traj[-1])+']'
        else:
            if '[' in traj[-1]:
                # example: ['61', '90', '50', '12[->37]'], after=50   
                # => 50[->12,->37]          
                side = ','.join([f"->{traj[-1][:traj[-1].find('[')]}"] + traj[-1][traj[-1].find('[')+1:-1].split(','))
                traj[-2]= f"{traj[-2]}[{side}]"                    
            else:
                traj[-2] = f"{traj[-2]}[->{traj[-1]}]"            
        traj.pop(-1)
        print("after", traj, after)
    else:
        traj.append(str(after))    
    return traj



def sample_walk(n, G, graph, model, all_nodes):
    N = len(G)     
    context = torch.zeros((1, N), dtype=torch.float64)
    start = graph.index_lookup[n]
    state = torch.zeros((1, len(G)), dtype=torch.float64)
    state[0, graph.index_lookup[n]] = 1.
    traj = [str(start)]    
    t = 0
    after = -1
    good = False   
    while True:      
        # print(f"input state {get_repr(state)}, context {get_repr(context)}, t {t}")  
        update, context = model(state, context, t)
        if not (state>=0).all():
            breakpoint()
        state = state_to_probs(state+update)
        t += 1
        state_numpy = state.detach().flatten().numpy()
        after = np.random.choice(len(state_numpy), p=state_numpy)        
        # try:
        #     print(f"post state {get_repr(state)}, context {get_repr(context)}, t {t}")
        #     print(f"sampled {after} with prob {state_numpy[after]}")
        # except:
        #     breakpoint()
        if ':' in all_nodes[after]:
            bad_ind = check_colon_order(all_nodes, traj, after)
            if bad_ind: break
        
        state = torch.zeros((1, len(G)), dtype=torch.float64)
        state[0, after] = 1.

        
        if extract(traj[-1]) == after:
            traj.append(str(after))
            break
        if after == start:
            traj.append(str(after))
            good = True
            break
        
        traj = append_traj(traj, after)
        if not traj:
            break


    return traj, good


def sample_walks(G, graph, walks, model, all_nodes, r_lookup, predefined_graph):       
    novel = []
    for _ in range(2):
        for n in G.nodes():
            if ':' in n: continue            
            traj, good = sample_walk(n, G, graph, model, all_nodes)  
            if len(traj) > 1 and good:
                name_traj = process_good_traj(traj, all_nodes)                
                assert len(traj) == len(name_traj)

                try:
                    root, edge_conn = verify_walk(r_lookup, predefined_graph, name_traj)
                    name_traj = '->'.join(name_traj)
                    print(name_traj, "success")
                    if name_traj not in walks:
                        print(name_traj, "novel")
                        walks.add(name_traj)
                        novel.append((name_traj, root, edge_conn))
                except:
                    print(name_traj, "failed")
    return novel


def main(args):
    walks = load_walks(args)
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
        if 'group-contrib' in args.motifs_folder:
            root_node, leaf_node, e = conn[-1]
            assert root_node.id == 0
            leaf_node.add_child((root_node, e)) # breaks dag
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
        setattr(args, 'log_folder', log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'config.json'), 'w+') as f:
            json.dump(json.dumps(args.__dict__), f)        
        model = diffuse(graph, log_dir, **diffusion_args)
    
    graph.reset()

    if args.diffusion_side_chains:
        layer = side_chain_grammar(index_lookup, args.log_folder)

    novel = sample_walks(G, graph, walks, model, all_nodes, r_lookup, predefined_graph)                

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
    main(args)