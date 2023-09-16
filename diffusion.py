import argparse
import pickle
import numpy as np
from utils.graph import *
import networkx as nx


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
    def __init__(self, dag, lookup):
        self.lookup = lookup
        self.dag = dag
        self.t = 0
        self.state = np.array([0 for _ in lookup])
        self.state[self.lookup[dag.val]] = 1
        self.frontier = {dag: 1}


    def step(self):
        new_frontier = defaultdict(float)
        for cur, p in self.frontier.items():
            if cur.side_chain:
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
            print(k, v, "iter")
            new_state[self.lookup[k.val]] = v
            print(v, new_state.sum())
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
    def __init__(self, dags, graph):
        self.dags = dags        
        self.t = 0        
        self.processes = []
        self.index_lookup = self.modify_graph(dags, graph)
        for dag in dags:
            self.processes.append(DiffusionProcess(dag, self.index_lookup))
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


    def step(self):
        for p in self.processes:
            p.step()
        self.t += 1
    
    def get_state(self):
        all_probs = [p.get_state() for (i, p) in enumerate(self.processes)]
        probs = np.array(all_probs).sum(axis=0)
        return probs/probs.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dags_file')
    parser.add_argument('--predefined_graph_file')
    args = parser.parse_args()

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

    graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
    graph = DiffusionGraph(dags, graph)
    print(f"state at 0: {graph.get_state()}")
    for t in range(10):
        graph.step()
        print(f"state at {t+1}: {graph.get_state()}")
        
        