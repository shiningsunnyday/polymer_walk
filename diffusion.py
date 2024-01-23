import argparse
import pickle
import numpy as np
from utils import *
import networkx as nx
import torch.nn as nn
import torch
import os
if 'TORCH_SEED' in os.environ:    
    seed = int(os.environ['TORCH_SEED'])
    torch.manual_seed(seed)
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import json
from copy import deepcopy
from math import factorial
from torch_geometric.nn.conv import GATConv, GCNConv, TransformerConv, GPSConv
from models.gnn_conv import MyGINConv as GINConv


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
        if self.split:
            res = []
            self.dfs_walk(dag, res)
            self.dfs_order = res
            self.num_nodes = len(res)            
        else:
            self.dfs_dir = dfs_seed >= 0
            dfs_seed = abs(dfs_seed)
            self.dfs_seed = dfs_seed % self.total # 0 to X-1, where X := prod_(node in main chain) num_childs(node)            
            res = self.compute_dfs(dag)
            new_res = self.augment_walk_order(res, dfs_seed)
            if dfs_seed == 0:
                assert res == new_res
            self.dfs_order = new_res
            self.num_nodes = len(res)



    def augment_walk_order(self, res, dfs_seed):
        """
        We want to augment vanilla dfs order "res" with dfs_seed
        dfs_seed // self.total is which node to start in main chain
        dfs_dir is the direction to travel
        
        1) Fill indices with the corresponding indices of res
        belonging to main chain.
        2) For each consecutive main chain nodes a and b, flip the order
        of res between a and b if dfs_dir
        """
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
            # we might loop back
            if indices[end] <= indices[start]:
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
    def impose_order(cur, add_main=False):
        for j, c in enumerate(cur.children):
            cur.children[j][0].side_chain = True               
        if add_main and len(cur.children):
            lowest_index_i = 0
            for j, c in enumerate(cur.children):
                if c[0].id < cur.children[lowest_index_i][0].id:
                    lowest_index_i = j     
            cur.children[lowest_index_i][0].side_chain = False
            DiffusionProcess.impose_order(cur.children[lowest_index_i][0], True)
        else:
            lowest_index_i = -1
        # make the rest side chain
        for j, c in enumerate(cur.children):
            if j != lowest_index_i:
                DiffusionProcess.impose_order(cur.children[j][0], False)            

    @staticmethod
    def compute_main_chain(dag):
        """
        we linearize a dag by canonicalizing the descendents
        for Group Contribution, this is done already because of .side_chain during construction
        for other datasets, we need to impose an ordering
        by default, just use the id
        """     
        dfs_order = []        
        need_impose = False
        DiffusionProcess.dfs_walk(dag, dfs_order)
        for cur in dfs_order:
            num_main_childs = sum([not c[0].side_chain for c in cur.children])
            if num_main_childs > 1:
                need_impose = True
        if need_impose:
            DiffusionProcess.impose_order(dag, True)
        
        chain = [dag]
        i = 0
        while len(chain) == 1 or chain[-1].id:
            i += 1        
            num_main_childs = sum([not c[0].side_chain for c in chain[-1].children])
            assert num_main_childs <= 1
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
                try:
                    DiffusionProcess.dfs_walk(c[0], res, perm_map)
                except:
                    breakpoint()
                if side_chain:
                    for i in range(len(res)-2, ind-1, -1):
                        res.append(res[i])
                    # print([a.val for a in res[:ind+1]], [a.val for a in res], "before after")


    
    def reset(self):
        self.t = 0
        self.state = np.array([0.0 for _ in self.lookup])
        if self.dag.val in self.lookup:
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

        # account for all non-single-node dags
        self.index_lookup = self.modify_graph(dags, graph)              
        for dag in dags:
            if dag.id:
                breakpoint()
            self.processes.append(DiffusionProcess(dag, self.index_lookup, **diffusion_args))
        self.graph = graph
        self.adj = nx.adjacency_matrix(graph).toarray()


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
            
        """
        Add nodes of the form GX:1, GX:2, ... representing re-visits to group X in the same walk
        """
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
            E = torch.rand((N, N), dtype=torch.float64)
        self.A = torch.as_tensor(diff_args['adj_matrix'], dtype=torch.float64)
        self.E = nn.Parameter(E)
        self.scale = nn.Parameter(torch.ones((self.A.shape[0],), dtype=torch.float64))        
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
            # W_new is (M, N, N)
            # self.A[None] is (1, N, N)
            W_hat = torch.matmul(W_new,self.A[None]) # (M, N, N)
            W_hat = W_hat.sum(axis=-2) # (M, N)
            W_hat_diag = torch.diag_embed(W_hat) # (M, N, N)
            L = W_hat_diag-W_new # (M, N, N)
            L = L/L.sum(axis=-1, keepdim=True)
        
        context = context*t/(t+1) + X/(t+1)        
        L_T = self.scale[None,...,None] * torch.transpose(L, -1,-2)
        update = torch.matmul(X[...,None,:],L_T).squeeze() # sum to 0
        return update, context
    

class Predictor(nn.Module):
    def __init__(self,
                 input_dim=16, 
                 hidden_dim=16, 
                 num_layers=5, 
                 num_heads=2, 
                 gnn='gin', 
                 edge_weights=False,
                 act='relu', 
                 share_params=True,
                 in_mlp=False,
                 mlp_out=False,
                 dropout_rate=0,
                 num_transformer_heads=1,
                 init='normal',
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gnn = gnn
        self.edge_weights = edge_weights
        self.in_mlp = in_mlp
        self.do_mlp_out = mlp_out
        self.share_params = share_params
        self.dropout_rate = dropout_rate
        self.num_transformer_heads = num_transformer_heads
        self.init = init
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        else:
            raise
        # assert input_dim == hidden_dim          
        if self.in_mlp:
            for i in range(1, num_heads+1):
                if share_params and i < num_heads:
                    continue
                lin_out_1 = nn.Linear(input_dim, hidden_dim)
                lin_out_2 = nn.Linear(hidden_dim, hidden_dim)
                if self.dropout_rate:
                    dropout = nn.Dropout(self.dropout_rate)
                    mlp_in = nn.Sequential(lin_out_1, act, dropout, lin_out_2)
                else:
                    mlp_in = nn.Sequential(lin_out_1, act, lin_out_2)
                if self.init == 'normal':
                    nn.init.normal_(lin_out_1.weight)
                    nn.init.normal_(lin_out_1.bias)
                    nn.init.normal_(lin_out_2.weight)
                    nn.init.normal_(lin_out_2.bias)
                elif self.init == 'zeros':
                    nn.init.zeros_(lin_out_1.weight)
                    nn.init.zeros_(lin_out_1.bias)
                    nn.init.zeros_(lin_out_2.weight)
                    nn.init.zeros_(lin_out_2.bias)                    
                layer_name = f"in_mlp" if share_params else f"in_mlp_{i}"
                setattr(self, layer_name, mlp_in)


        if self.do_mlp_out:
            for i in range(1, num_heads+1):
                if share_params and i < num_heads:
                    continue
                lin_out_1 = nn.Linear(hidden_dim, hidden_dim)
                lin_out_2 = nn.Linear(hidden_dim, hidden_dim)
                if self.dropout_rate:
                    dropout = nn.Dropout(self.dropout_rate)
                    mlp_out = nn.Sequential(lin_out_1, act, dropout, lin_out_2)
                else:
                    mlp_out = nn.Sequential(lin_out_1, act, lin_out_2)
                if self.init == 'normal':
                    nn.init.normal_(lin_out_1.weight)
                    nn.init.normal_(lin_out_1.bias)
                    nn.init.normal_(lin_out_2.weight)
                    nn.init.normal_(lin_out_2.bias)
                elif self.init == 'zeros':
                    nn.init.zeros_(lin_out_1.weight)
                    nn.init.zeros_(lin_out_1.bias)
                    nn.init.zeros_(lin_out_2.weight)
                    nn.init.zeros_(lin_out_2.bias)                     
                layer_name = f"mlp_out" if share_params else f"mlp_out_{i}"
                setattr(self, layer_name, mlp_out)
        
        if gnn == 'gin':
            for j in range(1, num_heads+1):
                input_dim = hidden_dim if self.in_mlp else self.input_dim
                if share_params and j < num_heads:
                    continue                
                for i in range(1, num_layers+1):
                    if i > 1: 
                        input_dim = hidden_dim     
                    lin_i_1 = nn.Linear(input_dim, hidden_dim)
                    lin_i_2 = nn.Linear(hidden_dim, hidden_dim)                        
                    if self.init == 'normal':
                        nn.init.normal_(lin_i_1.weight)
                        nn.init.normal_(lin_i_1.bias)
                        nn.init.normal_(lin_i_2.weight)
                        nn.init.normal_(lin_i_2.bias)                                   
                    elif self.init == 'zeros':
                        nn.init.zeros_(lin_i_1.weight)
                        nn.init.zeros_(lin_i_1.bias)
                        nn.init.zeros_(lin_i_2.weight)
                        nn.init.zeros_(lin_i_2.bias)                         
                    # setattr(self, f"gnn_{i}", GATConv(in_channels=-1, out_channels=hidden_dim, edge_dim=1))
                    if self.dropout_rate:
                        dropout = nn.Dropout(self.dropout_rate)
                        mlp = nn.Sequential(lin_i_1, act, dropout, lin_i_2)
                    else: 
                        mlp = nn.Sequential(lin_i_1, act, lin_i_2)
                    layer_name = f"gnn_{i}" if share_params else f"gnn_{i}_{j}"
                    setattr(self, layer_name, GINConv(mlp, edge_dim=1))               
        elif gnn in ['gat', 'gcn']:
            layer_name = {'gat': GATConv, 'gcn': GCNConv}
            if share_params:
                setattr(self, f"gnn_conv", layer_name[gnn](input_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=hidden_dim))
            else:
                for j in range(1, num_heads+1):
                    input_dim = hidden_dim if self.in_mlp else self.input_dim
                    setattr(self, f"gnn_conv_{j}", layer_name[gnn](input_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=hidden_dim))
        elif gnn in ['transformerconv', 'gpsconv']:
            for j in range(1, num_heads+1):
                input_dim = hidden_dim if self.in_mlp else self.input_dim
                if share_params and j < num_heads:
                    continue
                for i in range(1, num_layers+1):
                    if i > 1: 
                        input_dim = hidden_dim              
                    layer_name = f"gnn_{i}" if share_params else f"gnn_{i}_{j}"
                    if gnn == 'transformerconv':
                        setattr(self, layer_name, TransformerConv(input_dim, hidden_dim//self.num_transformer_heads, heads=self.num_transformer_heads, dropout=self.dropout_rate))
                    elif gnn == 'gpsconv':                
                        conv = TransformerConv(input_dim, hidden_dim//self.num_transformer_heads, heads=self.num_transformer_heads, dropout=self.dropout_rate)
                        setattr(self, layer_name, GPSConv(input_dim, conv=conv, heads=self.num_transformer_heads, dropout=self.dropout_rate))
                            
        else:
            raise NotImplementedError
                      

        for i in range(1, num_heads+1):
            lin_out_1 = nn.Linear(hidden_dim, hidden_dim)
            lin_out_2 = nn.Linear(hidden_dim, 1)
            if self.dropout_rate:
                dropout = nn.Dropout(self.dropout_rate)
                head_layers = [lin_out_1, act, dropout, lin_out_2]                
            else:
                head_layers = [lin_out_1, act, lin_out_2]
            mlp_out = nn.Sequential(*head_layers)
            if self.init == 'normal':
                nn.init.normal_(lin_out_1.weight)
                nn.init.normal_(lin_out_1.bias)
                nn.init.normal_(lin_out_2.weight)
                nn.init.normal_(lin_out_2.bias)
            elif self.init == 'zeros':
                nn.init.zeros_(lin_out_1.weight)
                nn.init.zeros_(lin_out_1.bias)
                nn.init.zeros_(lin_out_2.weight)
                nn.init.zeros_(lin_out_2.bias)                 
            setattr(self, f"out_mlp_{i}", mlp_out)
        
        

    def forward(self, X, edge_index, edge_weights, return_feats=False):  
        # node_mask = torch.zeros((X.shape[0],))==1
        # # node_mask = torch.ones((X.shape[0], 1))
        # node_mask[edge_index.flatten()] = True
        if self.gnn == 'gin':
            head_outs = []
            for j in range(1, self.num_heads+1):
                if self.share_params and j < self.num_heads:
                    continue
                X_out = X.clone()
                if self.in_mlp:
                    X_out = getattr(self, "in_mlp" if self.share_params else f"in_mlp_{j}")(X_out)
                for i in range(1, self.num_layers+1):
                    layer_name = f"gnn_{i}" if self.share_params else f"gnn_{i}_{j}"
                    X_out = getattr(self, layer_name)(X_out, edge_index, edge_weight=(edge_weights if self.edge_weights else None))
                head_outs.append(X_out)
            if self.share_params:
                assert len(head_outs) == 1
                X = head_outs[0]
                if self.do_mlp_out:
                    X = getattr(self, "mlp_out")(X)
                props = [getattr(self, f"out_mlp_{i}")(X) for i in range(1,self.num_heads+1)] 
            else:
                assert len(head_outs) == self.num_heads
                props = []                
                for i in range(1,self.num_heads+1):
                    if self.do_mlp_out:
                        head_outs[i-1] = getattr(self, f"mlp_out_{i}")(head_outs[i-1])
                    props.append(getattr(self, f"out_mlp_{i}")(head_outs[i-1]))
        elif self.gnn in ['transformerconv', 'gpsconv']:
            head_outs = []
            for j in range(1, self.num_heads+1):
                if self.share_params and j < self.num_heads:
                    continue
                X_out = X.clone()
                if self.in_mlp:
                    X_out = getattr(self, "in_mlp" if self.share_params else f"in_mlp_{j}")(X_out)
                for i in range(1, self.num_layers+1):
                    layer_name = f"gnn_{i}" if self.share_params else f"gnn_{i}_{j}"
                    X_out = getattr(self, layer_name)(X_out, edge_index)
                head_outs.append(X_out)
            if self.share_params:
                assert len(head_outs) == 1
                X = head_outs[0]
                if self.do_mlp_out:
                    X = getattr(self, "mlp_out")(X)
                props = [getattr(self, f"out_mlp_{i}")(X) for i in range(1,self.num_heads+1)] 
            else:
                assert len(head_outs) == self.num_heads
                props = []                
                for i in range(1,self.num_heads+1):
                    if self.do_mlp_out:
                        head_outs[i-1] = getattr(self, f"mlp_out_{i}")(head_outs[i-1])
                    props.append(getattr(self, f"out_mlp_{i}")(head_outs[i-1]))                    
        elif self.gnn in ['gat', 'gcn']:
            if self.share_params:
                if self.in_mlp:
                    X = getattr(self, "in_mlp" if self.share_params else f"in_mlp_{j}")(X)                
                
                X = getattr(self, f"gnn_conv")(X, edge_index, edge_weight=(edge_weights if self.edge_weights else None))
                if self.do_mlp_out:
                    X = getattr(self, "mlp_out")(X)                
                props = [getattr(self, f"out_mlp_{i}")(X) for i in range(1,self.num_heads+1)]
            else:
                head_outs = []
                for j in range(1, self.num_heads+1):   
                    X_out = X.clone()  
                    if self.in_mlp:
                        X_out = getattr(self, "in_mlp" if self.share_params else f"in_mlp_{j}")(X_out)                       
                    X_out = getattr(self, f"gnn_conv_{j}")(X_out, edge_index, edge_weight=(edge_weights if self.edge_weights else None))          
                    head_outs.append(X_out)
                props = []                
                for i in range(1,self.num_heads+1):
                    if self.do_mlp_out:
                        head_outs[i-1] = getattr(self, f"mlp_out_{i}")(head_outs[i-1])
                    props.append(getattr(self, f"out_mlp_{i}")(head_outs[i-1]))
        else:
            raise

        out = torch.cat(props, dim=-1)
        feats = X if self.share_params else head_outs
        if return_feats:
            return out, feats
        else:
            return out
    

def state_to_probs(state, adj=None):
    state = torch.where(state>=0., state, 0.0)
    if adj is not None:
        state[:, adj==0.] = 0.
    if state.sum(axis=-1) > 0:
        # return F.softmax(state)
        return state/state.sum(axis=-1)
    else:
        return state
        # print(f"all probs 0")
        # if adj is not None:
        #     state[:, adj!=0.] = 1.
        #     return state/state.sum(axis=-1)
        # else:
        #     breakpoint()
    state = state - state.min(-1, True).values
    return state/state.sum(axis=-1)


def walk_edge_weight(dag, graph, model, proc, eps=1e-6):
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
        state = state_to_probs(state+update, graph.adj[prev_node_ind])
        # print(f"post state {get_repr(state)}, context {get_repr(context)}, t {t}")  
        # dist = Categorical(state)
        # log_prob = dist.log_prob(cur_node_ind)
        t += 1
        W_adj[prev_node_ind, cur_node_ind] = max(state[0, cur_node_ind], eps)
        W_adj[cur_node_ind, prev_node_ind] = max(state[0, cur_node_ind], eps)
        # print(f"recounted {cur_node_ind} with prob {state[0, cur_node_ind]}")        
        state = torch.zeros((1, N), dtype=torch.float64)
        state[0, cur_node_ind] = 1.
        prev_node_ind = cur_node_ind  
    return W_adj 



def featurize_walk(args, graph, model, dag, proc, mol_feats, feat_lookup={}):
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
    if diff_args['opt'] == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=diff_args['alpha'])
    elif diff_args['opt'] == 'sgd':
        opt = torch.optim.Adam(model.parameters(), lr=diff_args['alpha'])
    else:
        raise
    history = []
    T = 10
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
        ax.set_yscale('log')
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
    """
    For example, what if we have G2->G2:1->G2:1?
    Or G2->G2? These are both cases of bad colon order, and should be avoided by design.
    """   
    if ':' in all_nodes[after]:
        ind = int(all_nodes[after].split(':')[-1])
    else:
        ind = 0
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
        if len(occur) == 1 or occur.sum() != 1: 
            return []    
        if len(traj) < 2 or str(after) != traj[-2].split('[')[0]: 
            return []
        # print("before", traj, after)
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
        # print("after", traj, after)
    else:
        traj.append(str(after))    
    return traj



def sample_walk(n, G, graph, model, all_nodes, loop_back=True):
    N = len(G)     
    context = torch.zeros((1, N), dtype=torch.float64)
    start = graph.index_lookup[n]
    state = torch.zeros((1, len(G)), dtype=torch.float64)
    state[0, graph.index_lookup[n]] = 1.
    traj = [str(start)]
    cur_node_ind = start    
    t = 0
    after = -1
    good = False   
    while True:      
        # print(f"input state {get_repr(state)}, context {get_repr(context)}, t {t}")  
        update, context = model(state, context, t)
        if not (state>=0).all():
            breakpoint()                         
        state = state_to_probs(state+update, graph.adj[cur_node_ind])
        state_numpy = state.detach().flatten().numpy()
        for i in range(len(G)):
            if len(traj) and extract(traj[-1]) == i: # can't loop back to itself if nothing else in between
                state_numpy[i] = 0.
            if check_colon_order(all_nodes, traj, i):
                state_numpy[i] = 0.
                        
        if state_numpy.max() <= 0.1: # set a threshold >= 0.
            if not loop_back:
                good = True
            break    
        t += 1
           
        state_numpy = state_numpy/state_numpy.sum()
        after = np.random.choice(len(state_numpy), p=state_numpy)        
        cur_node_ind = after
        # try:
        #     print(f"post state {get_repr(state)}, context {get_repr(context)}, t {t}")
        #     print(f"sampled {after} with prob {state_numpy[after]}")
        # except:
        #     breakpoint()                    
        state = torch.zeros((1, len(G)), dtype=torch.float64)
        state[0, after] = 1.

        # self-loop, not allowed!
        if extract(traj[-1]) == after:
            breakpoint()
            traj.append(str(after))
            break
        
        # loop back, done!
        if loop_back and after == start:
            traj.append(str(after))
            good = True
            break
    
        traj_after = append_traj(traj, after)
        if loop_back and not traj_after:
            break
        if not loop_back and not traj_after:
            good = True
            break

    return traj, good


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


def sample_walks(args, G, graph, seen_dags, model, all_nodes, r_lookup, predefined_graph, predict_args={}):
    new_dags = []
    trajs = []
    novel = []     
 
    num_tried, num_valid = 0, 0
    new_novel = 0
    index = 0

    if predict_args:
        assert 'predictor' in predict_args
        assert 'diffusion_args' in predict_args
        assert 'mol_feats' in predict_args
        assert 'feat_lookup' in predict_args
        predictor = predict_args['predictor']
        diffusion_args = predict_args['diffusion_args']
        mol_feats = predict_args['mol_feats']
        feat_lookup = predict_args['feat_lookup']

    
    while len(novel) < args.num_generate_samples:                    
        n = list(G.nodes())[index%len(G)]
        if ':' in n: continue
        traj, good = sample_walk(n, G, graph, model, all_nodes, loop_back='group-contrib' in os.environ['dataset'])
        if not good:
            breakpoint()            
        if len(traj) > 1 and good: 
            num_tried += 1
            name_traj = process_good_traj(traj, all_nodes)                       
            assert len(traj) == len(name_traj)                    
            try: # test for validity
                root, edge_conn = verify_walk(r_lookup, predefined_graph, name_traj, loop_back='group-contrib' in os.environ['dataset'])
                DiffusionGraph.value_count(root, {}) # modifies edge_conn with :'s too
                name_traj = '->'.join(name_traj)
                trajs.append(name_traj)
                # print(name_traj, "success")                    
                if is_novel(seen_dags, root):
                    seen_dags.append(root)
                    new_dags.append(root)
                    print(name_traj, "novel")    
                    if predict_args:                 
                        proc = DiffusionProcess(root, graph.index_lookup, **diffusion_args)
                        node_attr, edge_index, edge_attr = featurize_walk(args, graph, model, root, proc, mol_feats, feat_lookup)
                        W_adj = walk_edge_weight(root, graph, model, proc)
                        X = node_attr
                        prop = do_predict(predictor, X, edge_index, edge_attr, cuda=args.cuda)
                        print("predicted prop", prop)
                        # probs = [W_adj[int(traj[i])][int(traj[(i+1)%len(traj)])] for i in range(len(traj))]
                        novel.append((name_traj, root, edge_conn, W_adj, prop.detach().numpy()))
                    new_novel += 1
                    # p (lambda W_adj,edge_conn,graph:[[a.id,a.val,b.id,b.val,e,W_adj[graph.index_lookup[a.val]][graph.index_lookup[b.val]].item(),W_adj[graph.index_lookup[b.val]][graph.index_lookup[a.val]].item()] for (a,b,e) in edge_conn])(W_adj,edge_conn,graph)                            
                else:
                    print(f"{name_traj} discovered")
            except Exception as e:
                breakpoint()
                print(e)
                continue
            num_valid += 1
        index += 1
        # print(f"add {new_novel} samples, now {len(novel)} novel samples, validity: {num_valid}/{num_tried}")
    return novel, new_dags, trajs



def load_dags(args):
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
    return dags    


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
        if root_node.children: # don't diffuse single-group dags
            dags.append(root_node)
        

    graph = DiffusionGraph(dags, graph, **diffusion_args) 
    G = graph.graph
    all_nodes = list(G.nodes())
    diffusion_args['adj_matrix'] = nx.adjacency_matrix(G).toarray()    
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

    dags = load_dags(args)
    seen_dags = deepcopy(dags)
    novel, new_dags, trajs = sample_walks(args, G, graph, seen_dags, model, all_nodes, r_lookup, predefined_graph)                

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
    parser.add_argument('--opt', dest='diffusion_opt', default='adam')
    parser.add_argument('--num_epochs', dest='diffusion_num_epochs', default=500, type=int)
    # sampling params
    parser.add_argument('--num_generate_samples', type=int, default=15)    
    args = parser.parse_args()
    main(args)