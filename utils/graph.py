from .preprocess import *
import numpy as np
from itertools import product, permutations
import random
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from rdkit.Chem import AllChem, DataStructs
from copy import deepcopy
from .walk import walk_enumerate_mols
# from retro_star.api import planner


"""FOR GROUP CONTRIBUTION and ALL DATASETS"""
dataset = os.environ['dataset']
mols = load_mols(f'{dataset}/all_groups')
annotate_extra(mols, f'{dataset}/all_groups/all_extra.txt')


class Node:
    def __init__(self, parent, children, val, id, side_chain=False):        
        self.parent = parent
        self.children = children
        self.val = val
        self.id = id
        self.side_chain = side_chain

    def add_child(self, child):
        self.children.append(child)


def list_red_members(mol):
    r = []
    for i, a in enumerate(mol.GetAtoms()):
        r_grp = a.GetProp('r_grp')

        if not r_grp: continue
        for j in r_grp.split('-'):
            while int(j) > len(r):
                r.append([])
            r[int(j)-1].append(i)
    return r



def enumerate_black_subsets(mol, k):
    b = [] 
    for i, a in enumerate(mol.GetAtoms()):
        if not a.GetBoolProp('r'):
            b.append(i)
    if k in [1,2,3,4]:
        return permutations(b, k)
    elif k in [5,6]:
        res = []
        for ring in mol.GetRingInfo().AtomRings():
            for a in ring:
                if a not in b: continue
            for i in range(len(ring)):
                res.append(list(ring[i:]+ring[:i]))
                res.append(list(reversed(res[-1])))
        return res
    else:
        return permutations(b, k)


def inc(x):
    return [a+1 for a in x]


def dfs(edges, vis, cur, inds, num):
    vis[cur] = 1
    for b in edges[cur]:     
        if b not in inds: continue
        if vis[b]: continue
        dfs(edges, vis, b, inds, num)



def connected(mol, inds):
    vis = [0 for _ in range(mol.GetNumAtoms())]
    assert len(inds)
    cur = inds[0]
    edges = [[b.GetIdx() for b in a.GetNeighbors()] for a in mol.GetAtoms()]
    dfs(edges, vis, cur, inds, len(inds))
    return sum(vis) == len(inds)


def mol_to_graph(mol, inds=[], r=False):
    if not inds:
        inds = list(range(mol.GetNumAtoms()))
    graph = nx.Graph()
    for i, ind in enumerate(inds):
        a = mol.GetAtomWithIdx(ind)
        kwargs = {}
        if r: kwargs['r'] = a.GetBoolProp('r')
        graph.add_node(str(i), symbol=a.GetSymbol(), **kwargs)
    for i, u in enumerate(inds):
        for j, v in enumerate(inds):
            bond = mol.GetBondBetweenAtoms(u, v)
            if bond:
                graph.add_edge(str(i), str(j), bond_type=bond.GetBondType())
    return graph


def dag_to_graph(dag):
    graph = nx.DiGraph()
    bfs = [dag]
    graph.add_node(0, val=dag.val)
    while bfs:
        cur = bfs[0]
        bfs = bfs[1:]
        for j in cur.children:
            if isinstance(j, tuple):
                j = j[0]
            if j.id:
                graph.add_node(j.id, val=j.val.split(':')[0])
                bfs.append(j)
    return graph


def dag_isomorphic(dag1, dag2):
    def node_match(a, b):
        return a['val'] == b['val']
    g1 = dag_to_graph(dag1)
    g2 = dag_to_graph(dag2)
    return nx.isomorphism.is_isomorphic(g1, g2, node_match)
    


def check_order(orig_mol, mol, cls, r=False):
    """
    Check whether subgraph induced by cls ~ mol
    r further checks if red atoms are the same
    """    
    
    graph_1 = mol_to_graph(orig_mol, cls, r=r)
    graph_2 = mol_to_graph(mol, list(range(mol.GetNumAtoms())), r=r)

    if len(graph_1) != len(graph_2):
        return False
    if len(graph_1) == 1:
        breakpoint()
        res = list(dict(graph_1.nodes(data=True)).values())[0] == list(dict(graph_2.nodes(data=True)).values())[0]
    else:
        res = nx.is_isomorphic(graph_1, graph_2, node_match, edge_match)
    
    return res


def check_isomorphic(mol1, mol2, v1, v2):        
    """
    Strictly speaking, node-level isomorphism is only possible if the black and red atoms
    have the same atomic number. However, edge-level isomorphism is correct here. We have
    oxygen atoms replace carbon atoms, etc. but the bonds need to be consistent to ensure validity.
    """    
    # debug = False
    # if v1 == [14, 15, 4, 3, 2, 1, 0, 5] and v2 == [6, 9, 0, 1, 2, 3, 4, 5]:
    #     breakpoint()
    #     debug = True
    if len(v1) != len(v2): 
        return False   
    for i in range(len(v1)):
        a1 = mol1.GetAtomWithIdx(v1[i])
        a2 = mol2.GetAtomWithIdx(v2[i])
        if a1.GetBoolProp('r') or a2.GetBoolProp('r'):
            continue
        if a1.GetSymbol() != a2.GetSymbol():
            return False
    for i in range(len(v1)):
        for j in range(i+1, len(v1)):
            b1 = mol1.GetBondBetweenAtoms(v1[i], v1[j])
            b2 = mol2.GetBondBetweenAtoms(v2[i], v2[j])                   
            if (b1 == None) != (b2 == None): 
                return False
            # if b1 != None and b1.GetBondType() != b2.GetBondType():
            if b1 != None and abs(b1.GetBondTypeAsDouble()-b2.GetBondTypeAsDouble()) > 0.5:
                """
                We must enforce the bond types to be the same.
                """
                # if b1.GetBeginAtom().GetBoolProp('r'): continue
                # if b2.GetBeginAtom().GetBoolProp('r'): continue
                # if b1.GetEndAtom().GetBoolProp('r'): continue
                # if b2.GetEndAtom().GetBoolProp('r'): continue
                return False
    return True


def __extract_subgraph(mol, selected_atoms):
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetIntProp('org_idx', atom.GetIdx())

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots


def extract_subgraph(mol, selected_atoms): 
    subgraph_mapped, roots = __extract_subgraph(mol, selected_atoms) 
    subgraph = Chem.MolToSmiles(subgraph_mapped)
    subgraph = Chem.MolFromSmiles(subgraph)
    if subgraph is not None:
        return subgraph, subgraph_mapped, roots
    else:
        return None, subgraph_mapped, None
    


def substruct_matches(mol, subgraph):
    mol_copy = Chem.Mol(mol)
    res = list(map(list, mol_copy.GetSubstructMatches(subgraph)))
    if not res:
        # Chem.Kekulize(mol_copy)
        # res = list(map(list, mol_copy.GetSubstructMatches(subgraph)))
        try:
            suppl = Chem.ResonanceMolSupplier(mol_copy, Chem.KEKULE_ALL)
        except:
            return res
        for supp in suppl:
            if supp is None:
                continue
            res += list(map(list, supp.GetSubstructMatches(subgraph)))
    return res
    

def find_isomorphic(mol1, mol2, r_grp_1, rdkit=False):
    b2s = []
    exception_case = False
    if len(r_grp_1) == 1:
        exception_case = True
    if rdkit and not exception_case:
        mapped_subgraph = extract_subgraph(mol1, r_grp_1)[1]  
        b2s_new = substruct_matches(mol2, mapped_subgraph)
        """
        to facilitate the guessing, we can pick any subset of rings in b2
        and reverse them
        """
        # length = len(b2s_new)        
        # for i in range(length):
        #     b2 = b2s_new[i]
        #     ring_atoms = [r for ring in mol2.GetRingInfo().AtomRings() for r in ring]
        #     b2_index = [i for i in range(len(b2)) if b2[i] not in ring_atoms]
        #     if len(b2_index) > 4:
        #         continue
        #     for permute_nong_ring in permutations(b2_index):
        #         permute_nong_ring = list(permute_nong_ring)
        #         b2_permute = np.array(deepcopy(b2))
        #         b2_permute[permute_nong_ring] = b2_permute[b2_index]
        #         b2s_new.append(b2_permute.tolist())            
        # length = len(b2s_new) 
        # for i in range(length):
        #     b2 = b2s_new[i]
        #     b2_indices = []
        #     for ring in mol2.GetRingInfo().AtomRings():
        #         try:
        #             b2_index = [b2.index(r) for r in ring]
        #             b2_indices.append(b2_index)
        #         except:
        #             continue
        #     for j in range(1, 2**len(b2_indices)):
        #         mask = list(map(bool, map(int, format(j,'b'))))
        #         mask = [0 for _ in range(len(b2_indices)-len(mask))] + mask
        #         b2_copy = np.array(deepcopy(b2))
        #         for b2_index in np.array(b2_indices)[mask].tolist():
        #             stuff = b2_copy[sorted(b2_index)]
        #             stuff = list(reversed(stuff[1:].tolist()+stuff[:1].tolist())) # go in opposite direction
        #             b2_copy[sorted(b2_index)] = stuff 
        #             b2s_new.append(b2_copy.tolist())        
        # for b2 in b2s_new:
        #     # if len(b2) <= 8:
        #     #     b2s += [list(x) for x in permutations(b2)]
        #     # else:
        #     for i in range(len(b2)):
        #         b2s.append(list(b2[i:]+b2[:i]))
        #         b2s.append(list(reversed(b2s[-1]))) 
        #         b2s.append(list(list(reversed(b2[:i]))+b2[i:]))               
        #         b2s.append(list(b2[:i])+list(reversed(b2[i:])))
        #         b2s.append(list(list(reversed(b2[i:]))+b2[:i]))
        #         b2s.append(list((b2[i:]+list(reversed(b2[:i])))))
        # return b2s
        for b2 in b2s_new:
            G1 = mol_to_graph(mol2, b2)
            G2 = mol_to_graph(mol2, b2)
            gm = GraphMatcher(G1, G2, lambda a1, a2: a1['symbol']==a2['symbol'], lambda e1, e2: e1['bond_type']==e2['bond_type'])
            for lookup in gm.isomorphisms_iter():
                b2_new = deepcopy(b2)
                for i, n in enumerate(b2):
                    b2_new[i] = b2[int(lookup[str(i)])]
                b2s.append(b2_new)
    else:
        for b2 in enumerate_black_subsets(mol2, len(r_grp_1)):
            b2 = list(b2)
            if len(r_grp_1) != len(b2): continue
            if check_isomorphic(mol1, mol2, r_grp_1, b2): 
                b2s.append(b2)
    return b2s


def red_isomorphic(m1, m2, r_grp_1, r_grp_2, rdkit=False):
    # if m1 == 30 and m2 == 6 and r_grp_1 == [16]:
    #     breakpoint()    
    # if m1 == 57 and m2 == 59:
    #     breakpoint()
    # if m1 == 219 and m2 == 222:
    #     breakpoint()
    b2s = find_isomorphic(mols[m1-1], mols[m2-1], r_grp_1, rdkit=rdkit)
    # if not b2s:
    #     if r_grp_1: 
    #         return []
    #     else:
    #         b2s = [[]]
    b1s = find_isomorphic(mols[m2-1], mols[m1-1], r_grp_2, rdkit=rdkit)
    # if not b1s:
    #     if r_grp_2: 
    #         return []
    #     else:
    #         b1s = [[]]
    res = []
    conn_b1s = []
    try:
        for b1 in b1s:
            if connected(mols[m1-1], r_grp_1+b1):
                conn_b1s.append(b1)
    except AssertionError:
        print(m1, m2, "assert error")
        raise
    conn_b2s = []
    for b2 in b2s:
        if connected(mols[m2-1], b2+r_grp_2):
            conn_b2s.append(b2)
    for b2 in conn_b2s:     
        for b1 in conn_b1s:
            if (b1+b2) and check_isomorphic(mols[m1-1], mols[m2-1], r_grp_1+b1, b2+r_grp_2):
                return [r_grp_1, b1, b2, r_grp_2] # r_grp_1 <-> b2
    return res


def reds_isomorphic(m1, m2, rdkit=False):
    # if m1 == 57 and m2 == 59:
    #     breakpoint()
    # if name_group(m1)== 'G30' and name_group(m2) == 'G6':
    #     breakpoint()
    # if m1 == 219 and m2 == 222:
    #     breakpoint()
    # if m1 == 176 and m2 == 177:
    #     breakpoint()
    print(name_group(m1), name_group(m2))
    r_grp_m1 = list_red_members(mols[m1-1])
    r_grp_m2 = list_red_members(mols[m2-1])
    parallel = []
    # if rdkit and not r_grp_m1:
    #     r_grp_m1 = [[]]
    # if rdkit and not r_grp_m2:
    #     r_grp_m2 = [[]]
    for r_grp_1, r_grp_2 in product(r_grp_m1, r_grp_m2):        

        res = red_isomorphic(m1, m2, r_grp_1, r_grp_2, rdkit=rdkit)
        if res:
            parallel.append(res)
    return parallel


grp = '|'.join([f'(?:{name_group(i)})' for i in range(1, 98)])
grp = f'(?:{grp})'
pat_chain = f'(?:->{grp})+'
pat_inner = f'\[(?:(?:,?){pat_chain})+\]'
pat_or = f'(?:({grp}{pat_inner})|({grp}))'
pat = f'(?:{pat_or})(?:->{pat_or})*'

def extract_chain(walk):
    side_chains = []
    while True:
        chain_search = re.search(pat_inner, walk)
        if not chain_search: break
        start, end = chain_search.span()
        side_chains.append(walk[start:end])
        walk = walk[:start]+'#'+walk[end:] 
    
    res = walk.split('->')
    j = 0
    for i in range(len(res)):
        if '#' in res[i]:
            res[i] = res[i].replace('#', side_chains[j])
            j += 1
    return res   

# string parsing approach
def chain_extract(walk, predefined_graph):
    def get_alpha(walk, i):
        j = i+1
        while j<len(walk):
            if not walk[j].isalnum():
                break
            j += 1
        return j-1
        
    graph = nx.DiGraph()
    prev_node = -1
    e = -1
    inner = False
    side = False
    i = 0   
    last_main_node = 0 
    while i < len(walk):
        if walk[i] == '-':
            j = i+1
            while j<len(walk):
                if walk[j] == '>':
                    break
                j += 1
            try:
                e = int(walk[i+1:j])
            except:
                breakpoint()
            i = j+1
        elif walk[i].isalnum():
            j = get_alpha(walk, i)
            if j+1 == len(walk): # last node
                new_node = 0
            else:
                new_node = len(graph)
            graph.add_node(new_node, val=walk[i:j+1], side=inner)
            if not inner:
                last_main_node = new_node
            if prev_node >= 0:
                try:
                    prev_val = graph.nodes[prev_node]['val']
                    new_val = graph.nodes[new_node]['val']
                    r_grp_1 = predefined_graph[prev_val][new_val][e]['r_grp_1']            
                    r_grp_2 = predefined_graph[prev_val][new_val][e]['r_grp_2']                    
                except:
                    breakpoint()

                graph.add_edge(prev_node, new_node, e=e, r_grp_1=r_grp_1, r_grp_2=r_grp_2)
            e = -1
            i = j+1
            if inner:
                side = True            
            if not inner or side:
                prev_node = new_node

        elif walk[i] == ',':
            side = False
            prev_node = last_main_node
            i += 1
        elif walk[i] == '[':
            inner = True            
            i += 1
        elif walk[i] == ']':
            inner = False
            side = False
            prev_node = last_main_node
            i += 1
        else:
            breakpoint()
            raise

    map = {}
    for node in graph.nodes(data=True):
        map[node[0]] = node[1]['val']
    for edge in graph.edges(data=True):        
        print(f"{map[edge[0]]}:{edge[0]}->{map[edge[1]]}:{edge[1]}, {edge[2]['e']}")
      
    return graph





def extract(x):
    return int(x.split('[')[0]) if '[' in x else int(x)


def extract_inner(x):
    assert '[' in x
    assert ']' in x
    return x[x.find('[')+1:x.find(']')]


def linearize(x):
    # A[->C->D] => A->C->D
    # A[->C] => A-.C
    assert ',' not in x
    pass



def search_walk(i, walk, graph, cur):
    if i == len(walk)-1: return cur
    breakpoint()
    a = walk[i]
    b = walk[i+1]
    for e in graph[a][b]:
        cur.append(e)
        new_cur = search_walk(i+1, walk, graph, cur)
        if new_cur: return new_cur
        cur.pop()
    return []


def find_e(graph, a, b, e1):
    r_grp_1 = graph[a][b][e1]['r_grp_1']
    r_grp_2 = graph[a][b][e1]['r_grp_2']
    e2s = []
    for e2, e2_data in graph[b][a].items():
        if e2_data['r_grp_1'] == r_grp_2 and e2_data['r_grp_2'] == r_grp_1:
            e2s.append(e2)
    assert len(e2s) == 1
    return e2s[0]


def bfs_traverse(G, graph):
    """
    Difference with verify_walk is that G is a graph, instead of a walk
    By default, make all nodes side chain, then find longest path to make it main chain
    """
    paths = dict(nx.all_pairs_shortest_path(G))
    path_len = 0
    main_nodes = []
    for path_start, paths_k in paths.items():        
        for path_end, path in paths_k.items():
            if len(path) > path_len:
                path_len = len(path)
                start = path_start
                main_nodes = path
    prev_node = Node(None, [], start.split(':')[0], 0, side_chain=False)
    id = 1
    bfs = [(prev_node, start)]
    vis = {n: False for n in G.nodes()}
    vis[start] = True
    conn = []
    root_node = prev_node
    circ_edge = None
    while bfs:
        prev_node, prev_node_val = bfs.pop(0)
        is_leaf = True
        for cur in G[prev_node_val]:            
            if vis[cur]:
                continue
            assert len(G[prev_node_val][cur]) == 1
            r_grp_1 = G[prev_node_val][cur][0]['r_grp_1']
            r_grp_2 = G[prev_node_val][cur][0]['r_grp_2']
            i = find_edge(graph, prev_node_val.split(':')[0], cur.split(':')[0], r_grp_1, r_grp_2)
            # i = list(graph[prev_node_val][cur])[0]
            cur_node = Node((prev_node, i), [], cur.split(':')[0], id, side_chain=(cur not in main_nodes))
            id += 1
            prev_node.add_child((cur_node, i))
            assert len(list(G[prev_node_val][cur])) == 1
            # if i is None:
            #     breakpoint()
            conn.append((prev_node, cur_node, i))
            conn.append((cur_node, prev_node, i))
            vis[cur] = True
            bfs.append((cur_node, cur))
    if root_node.id:
        breakpoint()
    return root_node, conn   


def dfs_traverse(walk, loop_back=False):
    """
    walk can be a list, e.g. [L3, S32, S20[->S1,->S1], S32, L3]
    or a graph, e.g. 
        L3:0->S32:1, 2
        S32:1->S20:2, 2
        S20:2->S1:3, 1
        S20:2->S1:4, 4
        S20:2->S32:5, 2
        S32:5->L3:6, 2
    add, for every connected a, b
        (Node for a, Node for b)
        (Node for b, Node for a)
    to conn
    at the end, connect last leaf, root node and do the same
    return root node, conn
    """
    prev_node = None
    root_node = None
    conn = []
    id = 0     
    if loop_back and walk[len(walk)-1] != walk[0]: # come back to origin
        walk.append(walk[0])    
    if loop_back:
        walk_len = len(walk)-1 
    else:
        walk_len = len(walk)    
    for i in range(walk_len):
        if '[' in walk[i]:
            start = walk[i].find('[')
            assert ']' == walk[i][-1]
            grps = [walk[i][:start]]
            for grp in walk[i][start+1:-1].split(','):
                grps.append(grp[2:])
            cur = Node(prev_node, [], grps[0], id)
            id += 1
            if i: 
                conn.append((prev_node, cur))
                conn.append((cur, prev_node))
            for g in grps[1:]:
                prev_g = cur
                for next_g in g.split('->'):
                    g_child = Node(prev_g, [], next_g, id, True)
                    id += 1
                    prev_g.add_child(g_child)
                    conn.append((prev_g, g_child))
                    conn.append((g_child, prev_g))
                    prev_g = g_child                    
            if i:
                prev_node.add_child(cur)           
            prev_node = cur
        else:
            cur = Node(prev_node, [], walk[i], id)
            id += 1
            if i: 
                prev_node.add_child(cur)
                conn.append((prev_node, cur))
                conn.append((cur, prev_node))
            prev_node = cur

        root_node = root_node or prev_node

    if loop_back: # come back to origin
        conn.append((root_node, prev_node))
        conn.append((prev_node, root_node))

    return root_node, conn


def traverse_dfs(walk, graph, loop_back=False):
    conn = []
    start = list(walk.nodes)[0]
    vis = [False for n in walk.nodes()]
    nodes = [None for n in walk.nodes()]
    to_do = [start]
    root_node = Node(None, [], walk.nodes[start]['val'], 0)
    nodes[start] = root_node
    while to_do:
        cur = to_do.pop()
        vis[cur] = True
        neis = sorted((walk.nodes[j]['side'], j) for j in walk[cur])
        for side, j in neis:
            if vis[j]:
                if j:
                    continue
                elif loop_back: # cycle back to root
                    e = walk[cur][j]['e'] 
                    e_cyc = find_e(graph, nodes[cur].val, nodes[j].val, e)       
                    nodes[cur].add_child((nodes[j], e))
                    nodes[j].parent = (nodes[cur], e_cyc)
                    conn.append((nodes[j], nodes[cur], e_cyc))                    
                    break
            to_do.append(j)
            node = Node(nodes[cur], [], walk.nodes[j]['val'], j, side)
            nodes[j] = node    
            e = walk[cur][j]['e']        
            nodes[cur].add_child((nodes[j], e))
            nodes[j].parent = (nodes[cur], find_e(graph, nodes[cur].val, nodes[j].val, e))
            conn.append((nodes[cur], nodes[j], e))
    

    return root_node, conn
          

def find_edge(graph, a, b, r_grp_a, r_grp_b):
    for i in range(len(graph[a][b])):
        if graph[a][b][i]['r_grp_1'] != r_grp_a:
            continue
        if graph[a][b][i]['r_grp_2'] != r_grp_b:
            continue            
        return i      



def guess_edge_conn(graph, conn, r_lookup):
    num_tries = 100
    for _ in range(num_tries):
        edge_conn = []                
        used_reds = defaultdict(set)
        for a, b in conn:
            if a.id > b.id:
                continue
            bad = True
            e_inds = list(range(len(graph[a.val][b.val])))
            random.shuffle(e_inds)
            for i in e_inds:
                red_j1 = graph[a.val][b.val][i]['r_grp_1']
                red_j2 = graph[a.val][b.val][i]['r_grp_2']
                assert tuple(red_j1) in set(tuple(x) for x in r_lookup[a.val].values())
                assert tuple(red_j2) in set(tuple(x) for x in r_lookup[b.val].values())
                if set(red_j1) & used_reds[a.id]:
                    continue
                if set(red_j2) & used_reds[b.id]:
                    continue
                bad = False
                used_reds[a.id] |= set(red_j1)
                used_reds[b.id] |= set(red_j2)
                # print(f"{a.val}->{b.val}")
                # print(f"used {used_reds[a]}")
                edge_conn.append((a, b, i))
                if b in a.children:
                    ind = a.children.index(b)
                    a.children[ind] = (b, i)
                    assert b.parent == a
                    if find_edge(graph, a.val, b.val, red_j1, red_j2) != i:
                        breakpoint()
                    b.parent = (a, find_edge(graph, b.val, a.val, red_j2, red_j1))
                break
            if bad:
                break
        if not bad:
            break
    if bad:
        return None
    return edge_conn       



def verify_edge_conn(graph, conn, r_lookup):
    edge_conn = []                
    used_reds = defaultdict(set)
    for a, b, i in conn:
        if a.id > b.id:
            continue
        red_j1 = graph[a.val][b.val][i]['r_grp_1']
        red_j2 = graph[a.val][b.val][i]['r_grp_2']
        if not tuple(red_j1) in set(tuple(x) for x in r_lookup[a.val.split(':')[0]].values()):
            breakpoint()
        if not tuple(red_j2) in set(tuple(x) for x in r_lookup[b.val.split(':')[0]].values()):
            breakpoint()
        if set(red_j1) & used_reds[a.id]:
            continue
        if set(red_j2) & used_reds[b.id]:
            continue
        bad = False
        used_reds[a.id] |= set(red_j1)
        used_reds[b.id] |= set(red_j2)
        # print(f"{a.val}->{b.val}")
        # print(f"used {used_reds[a]}")
        edge_conn.append((a, b, i))
        if b in a.children:
            ind = a.children.index(b)
            a.children[ind] = (b, i)
            assert b.parent == a
            if find_edge(graph, a.val, b.val, red_j1, red_j2) != i:
                breakpoint()
            b.parent = (a, find_edge(graph, b.val, a.val, red_j2, red_j1))
    return edge_conn
 



def verify_walk(r_lookup, graph, walk, loop_back=False):
    # r_lookup: dict(red group id: atoms)
    # check all edges exist
    # test_walk = [('0', '1', 'L3', 'S32', 2), ('1', '2', 'S32', 'S20', 2), ('2', '5', 'S20', 'S32', 2), ('2', '3', 'S20', 'S1', 1), ('2', '4', 'S20', 'S1', 4)]
    # possible_connections(('5', 'S32'), graph, test_walk, r_lookup)
    try:         
        if isinstance(walk, nx.Graph):            
            root, edge_conn = traverse_dfs(walk, graph, loop_back=loop_back)            
            used_reds = defaultdict(set)
            for a, b, i in edge_conn:
                red_j1 = graph[a.val][b.val][i]['r_grp_1']
                red_j2 = graph[a.val][b.val][i]['r_grp_2']
                assert tuple(red_j1) in set(tuple(x) for x in r_lookup[a.val].values())
                assert tuple(red_j2) in set(tuple(x) for x in r_lookup[b.val].values())
                if set(red_j1) & used_reds[a.id]:
                    breakpoint()
                if set(red_j2) & used_reds[b.id]:
                    breakpoint()
                used_reds[a.id] |= set(red_j1)
                used_reds[b.id] |= set(red_j2)            
            
            print("pass!")
        elif isinstance(walk, list):
            root, conn = dfs_traverse(walk, loop_back=loop_back)       
            for a, b in conn:
                if b.val not in graph[a.val]:
                    raise KeyError(f"{a.val} {b.val} not connected")                            
            # edge_conn = guess_edge_conn(graph, conn[::-2], r_lookup)
            # if edge_conn is None:
            #     raise RuntimeError(walk)
            """
            tracks all possible connections, then prunes the isomorphic
            copies at each step
            """    
            walk = [(str(a.id), str(b.id), a.val.split(':')[0], b.val.split(':')[0]) for (a,b) in conn]
            chosen_edges, new_mol = walk_enumerate_mols(walk, graph, mols, loop_back=loop_back)            
            assert new_mol is not None
            root.smiles = Chem.MolToSmiles(new_mol)
            if Chem.MolFromSmiles(root.smiles) is None:
                raise ValueError(root.smiles)
            conn = [(a,b,e) for (a,b), e in zip(conn, chosen_edges)]
            edge_conn = verify_edge_conn(graph, conn, r_lookup)
        else: 
            raise NotImplementedError
    except Exception as e:        
        raise e
         
    return root, edge_conn



def is_novel(dags, root):
    for dag in dags:
        if dag_isomorphic(dag, root):
            return False
    return True
    

def mol2fp(mol,nBits=2048): 
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits)  
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr    


def possible_connections(cur, graph, edge_conn, r_lookup):
    used_reds = defaultdict(set)
    for a_id, b_id, a_val, b_val, i in edge_conn:
        red_j1 = graph[a_val][b_val][i]['r_grp_1']
        red_j2 = graph[a_val][b_val][i]['r_grp_2']
        assert tuple(red_j1) in set(tuple(x) for x in r_lookup[a_val].values())
        assert tuple(red_j2) in set(tuple(x) for x in r_lookup[b_val].values())
        if set(red_j1) & used_reds[a_id]:
            breakpoint()
        if set(red_j2) & used_reds[b_id]:
            breakpoint()
        used_reds[a_id] |= set(red_j1)
        used_reds[b_id] |= set(red_j2)            
    
    print("pass!")
    poss = []
    cur_id, cur_val = cur
    for dest in graph[cur_val]:   
        for e in graph[cur_val][dest]:
            r_grp = graph[cur_val][dest][e]['r_grp_1']
            if set(r_grp) & set(used_reds[cur_id]):
                continue
            poss.append((dest, e))
