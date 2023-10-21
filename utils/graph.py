from .preprocess import *
from itertools import product, permutations

mols = load_mols('/home/msun415/polymer_walk/data/all_groups')
# mols = load_mols('/home/msun415/polymer_walk/data/datasets/pu_groups/all_groups')
# annotate_extra(mols, 'data/datasets/pu_groups/all_groups/all_extra.txt')

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
    if k in [1,2,3,4,5]:
        return permutations(b, k)
    elif k == 6:
        res = []
        for ring in mol.GetRingInfo().AtomRings():
            for a in ring:
                if a not in b: continue
            for i in range(len(ring)):
                res.append(list(ring[i:]+ring[:i]))
                res.append(list(reversed(res[-1])))
        return res
    else:
        breakpoint()


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
    cur = inds[0]
    edges = [[b.GetIdx() for b in a.GetNeighbors()] for a in mol.GetAtoms()]
    dfs(edges, vis, cur, inds, len(inds))
    return sum(vis) == len(inds)



def check_isomorphic(mol1, mol2, v1, v2):
    debug = False
    # if v1 == [14, 15, 4, 3, 2, 1, 0, 5] and v2 == [6, 9, 0, 1, 2, 3, 4, 5]:
    #     breakpoint()
    #     debug = True
    if len(v1) != len(v2): return False
    for i in range(len(v1)):
        for j in range(i+1, len(v1)):
            try:
                b1 = mol1.GetBondBetweenAtoms(v1[i], v1[j])
                b2 = mol2.GetBondBetweenAtoms(v2[i], v2[j])
            except:
                breakpoint()
            if (b1 == None) != (b2 == None): 
                return False
    return True


def find_isomorphic(mol1, mol2, r_grp_1):
    b2s = []
    for b2 in enumerate_black_subsets(mol2, len(r_grp_1)):
        b2 = list(b2)
        if len(r_grp_1) != len(b2): continue
        if check_isomorphic(mol1, mol2, r_grp_1, b2): 
            b2s.append(b2)
    return b2s


def red_isomorphic(m1, m2, r_grp_1, r_grp_2):
    b2s = find_isomorphic(mols[m1-1], mols[m2-1], r_grp_1)
    if not b2s: return 0
    b1s = find_isomorphic(mols[m2-1], mols[m1-1], r_grp_2)
    if not b1s: return 0
    res = []
    for b2 in b2s:
        for b1 in b1s:
            if check_isomorphic(mols[m1-1], mols[m2-1], r_grp_1+b1, b2+r_grp_2):
                if connected(mols[m1-1], r_grp_1+b1) and connected(mols[m2-1], r_grp_2+b2):
                    return [r_grp_1, b1, b2, r_grp_2] # r_grp_1 <-> b2
    return res


def reds_isomorphic(m1, m2):
    print(name_group(m1), name_group(m2))
    r_grp_m1 = list_red_members(mols[m1-1])
    r_grp_m2 = list_red_members(mols[m2-1])
    parallel = []
    for r_grp_1, r_grp_2 in product(r_grp_m1, r_grp_m2):        
        res = red_isomorphic(m1, m2, r_grp_1, r_grp_2)
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


def extract(x):
    return int(x.split('[')[0]) if '[' in x else int(x)


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


def bfs_traverse(G):
    """
    differences with verify_walk:
    - 
    """
    start = list(G.nodes)[0]
    prev_node = Node(None, [], start.split(':')[0], 0)
    id = 1
    bfs = [prev_node]
    vis = {n: False for n in G.nodes()}
    vis[start] = True
    conn = []
    root_node = prev_node
    while bfs:
        prev_node = bfs.pop(0)
        for cur in G[prev_node.val]:            
            if vis[cur]: continue
            i = list(G[prev_node.val][cur])[0]
            cur_node = Node((prev_node, i), [], cur.split(':')[0], id)
            id += 1
            prev_node.add_child((cur_node, i))
            assert len(list(G[prev_node.val][cur])) == 1
            conn.append((prev_node, cur_node, i))
            conn.append((cur_node, prev_node, i))
            vis[cur] = True
            bfs.append(cur_node)

    return root_node, conn   


def dfs_traverse(walk):
    prev_node = None
    root_node = None
    conn = []
    id = 0
    for i in range(len(walk)):
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
                g_child = Node(cur, [], g, id, True)
                id += 1
                cur.add_child(g_child)
                conn.append((cur, g_child))
                conn.append((g_child, cur))

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

    conn.append((root_node, prev_node))
    conn.append((prev_node, root_node))

    return root_node, conn


def verify_walk(r_lookup, graph, walk):
    # r_lookup: dict(red group id: atoms)
    # check all edges exist
    def find_edge(red_key, dir, r_grps):
        for i, r_grp in r_grps.items():
            if dir == 'k1' and r_grp == red_key[dir][:len(r_grp)]:
                return r_grp
            if dir == 'k2' and r_grp == red_key[dir][-len(r_grp):]:
                return r_grp
        breakpoint()
    
    try:
        root, conn = dfs_traverse(walk)
    except:
        raise
    used_reds = defaultdict(set)
    edge_conn = []
    bad = False
    for a, b in conn:
        for i in range(len(graph[a.val][b.val])):            
            e = graph[a.val][b.val][i]
            red_j1 = find_edge(e, 'k1', r_lookup[a.val])
            if set(red_j1) & used_reds[a]: 
                if i == len(graph[a.val][b.val])-1: bad = True
                continue
            else: 
                used_reds[a] |= set(red_j1)
                # print(f"{a.val}->{b.val}")
                # print(f"used {used_reds[a]}")
                edge_conn.append((a, b, i))
                if b in a.children:
                    ind = a.children.index(b)
                    a.children[ind] = (b, i)
                elif b == a.parent:
                    a.parent = (b, i)
                break
        if bad: raise
     
    return root, edge_conn

