import argparse
import pickle
import numpy as np
from utils.graph import *
import networkx as nx
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import json
from torch.distributions import Categorical


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
    def __init__(self, dag, lookup, side_chains=False, **diffusion_args):
        self.lookup = lookup
        self.dag = dag
        self.side_chains = side_chains
        self.main_chain = []
        self.reset()


    @staticmethod
    def compute_main_chain(dag):
        chain = [dag]
        while len(chain) == 1 or chain[-1].id:
            for child, _ in chain[-1].children:
                if child.side_chain: continue
                chain.append(child)
                break
        return chain


    
    def reset(self):
        self.t = 0
        self.state = np.array([0.0 for _ in self.lookup])
        self.state[self.lookup[self.dag.val]] = 1.0
        self.frontier = {self.dag: 1}


    def step(self):
        new_frontier = defaultdict(float)
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
    
    

def diffuse(graph, log_folder, **diff_args):
    G = graph.graph
    print(f"state at 0: {graph.get_state()}")
    N, M = len(G), 1 if diff_args['combine_walks'] else len(graph.processes)
    if diff_args['e_init']:
        E = torch.as_tensor(nx.adjacency_matrix(G).toarray(), dtype=torch.float64)
    else:
        E = torch.zeros((N, N), dtype=torch.float64)
    scale = nn.Parameter(torch.ones((1,), dtype=torch.float64))
    W = nn.Parameter(E)
    A = torch.as_tensor(E.clone().detach(), dtype=torch.float64)
    context_layer = nn.Linear(N, N*N, dtype=torch.float64)
    nn.init.zeros_(context_layer.weight)
    nn.init.zeros_(context_layer.bias)
    loss_func = nn.MSELoss()   
    opt = torch.optim.SGD([W, scale]+list(context_layer.parameters()), lr=diff_args['alpha'])
    history = []
    T = 10
    for i in range(diff_args['num_epochs']):
        graph.reset()
        context = torch.zeros((M, N), dtype=torch.float64)
        for t in range(T):
            t_losses = []
            opt.zero_grad()             
            if diff_args['combine_walks']:
                L = torch.diag((W*A).sum(axis=0))-W # (N, N)
            else:
                if diff_args['context_L']:
                    adjust = context_layer(context).reshape((-1,)+W.shape)
                    W_new = adjust + W # (M, N, N)
                else:
                    W_new = W
                L = torch.diag_embed(torch.matmul(W_new,A[None]).sum(axis=-2))-W_new # (M, N, N)

            X = torch.as_tensor(graph.get_state(not diff_args['combine_walks'])) # (M, N)
            context = context*t/(t+1) + X/(t+1)
            graph.step()
            Y = torch.as_tensor(graph.get_state(not diff_args['combine_walks'])) # (M, N)            
            L_T = scale * torch.transpose(L, -1,-2)
            update = torch.matmul(X[...,None,:],L_T).squeeze()
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

    D = torch.diag((W*A).sum(axis=0))
    L = (D-W).detach().numpy()
    return L


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
    parser.add_argument('--predefined_graph_file')
    parser.add_argument('--log_folder')
    # diffusion args
    parser.add_argument('--side_chains', dest='diffusion_side_chains', action='store_true')
    parser.add_argument('--combine_walks', dest='diffusion_combine_walks', action='store_true')
    parser.add_argument('--e_init', dest='diffusion_e_init', action='store_true')
    parser.add_argument('--context_L', dest='diffusion_context_L', action='store_false')
    parser.add_argument('--alpha', dest='diffusion_alpha', default=1e-4, type=float)
    parser.add_argument('--num_epochs', dest='diffusion_num_epochs', default=500, type=int)

    args = parser.parse_args()
    diffusion_args = {k[len('diffusion_'):]: v for (k, v) in args.__dict__.items() if 'diffusion' in k}

    graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)

    num_nodes = len(graph.nodes())
    index_lookup = dict(zip(graph.nodes(), range(num_nodes)))
    data = pickle.load(open(args.dags_file, 'rb'))
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
    if args.log_folder:
        L = pickle.load(open(os.path.join(args.log_folder, 'L.pkl'), 'rb'))
    else:        
        log_dir = os.path.join('logs/', f'logs-{time.time()}/')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'config.json'), 'w+') as f:
            json.dump(json.dumps(args.__dict__), f)        
        L = diffuse(graph, log_dir, **diffusion_args)
        pickle.dump(L, open(os.path.join(log_dir, 'L.pkl'), 'wb+'))
    
    graph.reset()
    G = graph.graph
    all_nodes = list(G.nodes())
    trajs = []

    if not args.diffusion_side_chains:
        layer = side_chain_grammar(index_lookup, args.log_folder)

    for _ in range(1000):
        for n in G.nodes():
            start = graph.index_lookup[n]
            state = np.zeros(len(G))
            state[graph.index_lookup[n]] = 1.
            after = -1
            traj = [start]
            while after != start:
                state += L@state
                state = np.where(state<0, 0, state)
                state/=state.sum()
                after = np.random.choice(len(state), p=state)
                traj.append(after)
                if traj[-1] == traj[-2]: break
                state = np.zeros(len(G))
                state[after] = 1.
            
            if len(traj) > 1 and traj[-1] != traj[-2]:
                trajs.append(traj)
    
    trajs = sorted(trajs, key=lambda x: len(x))
    trajs = [[all_nodes[x] for x in traj] for traj in trajs]
    print(trajs)
    