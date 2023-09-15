from utils.preprocess import *
from itertools import product, permutations

mols = load_mols('/home/msun415/polymer_walk/data/all_groups')


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
    if k in [1,2,3]:
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
                    return [r_grp_1+b1, b2+r_grp_2]
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
