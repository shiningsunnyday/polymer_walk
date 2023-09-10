import argparse
from rdkit import Chem
from rdkit.Chem import Draw
import os
from itertools import product, permutations

def load_mols(folder):
    mols = []
    files = os.listdir(folder)
    files = list(filter(lambda x: x.endswith('.mol'), files))
    files = sorted(files, key=lambda x:int(x.rstrip('.mol')))

    for f in files:
        if f.endswith('.mol'):
            mol = Chem.rdmolfiles.MolFromMolFile(os.path.join(folder, f))
            mols.append(mol)
            for i, a in enumerate(mol.GetAtoms()): 
                a.SetProp("molAtomMapNumber", str(i+1))
    return mols


def annotate_extra(mols, path):
    all_labels = []
    labels = []
    volumes = []
    red_grps = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if line.startswith('Volume:'):
                volumes = line[len('Volume:'):].rstrip('\n').split(',')
                continue
            if not line:
                break
            if line == '\n':
                if labels:
                    all_labels.append(labels)
                    labels = []
            else:
                red_grp = ''
                label = line.split(':')[0]
                if ':' in line:
                    red_grp = line.split(':')[1]
                else:
                    red_grp = ''

                labels.append((int(label), red_grp.rstrip('\n')))

    print(len(mols), len(all_labels))
    if volumes: assert len(volumes) == len(mols)
    for i in range(len(mols)):
        m = mols[i]
        if volumes: m.GetConformer().SetProp('V', volumes[i])
        labels = [l[0] for l in all_labels[i]]
        red_grps = [l[1] for l in all_labels[i]]
        for i in range(m.GetNumAtoms()):
            m.GetAtomWithIdx(i).SetBoolProp('r', False)
            m.GetAtomWithIdx(i).SetProp('r_grp', '')
        for i in range(m.GetNumBonds()):
            m.GetBondWithIdx(i).SetBoolProp('r', False)
            m.GetBondWithIdx(i).SetBoolProp('picked', False)
        for l, red_grp in zip(labels, red_grps):
            if l <= m.GetNumAtoms():
                a = m.GetAtomWithIdx(l-1)
                a.SetBoolProp('r', True)
                a.SetProp('r_grp', red_grp)
                a.SetProp('atomLabel', f"R_{a.GetSymbol()}{l}")
                for b in a.GetBonds():
                    b.SetBoolProp('r', True)
    return red_grps

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
    if k in [1,2]:
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


def red_isomorphic(mols, m1, m2):
    prefix = lambda x: "P" if x <= 41 else ("S" if x <= 73 else "L")
    num = lambda m: prefix(m)+f"{m if m<=41 else m-41 if m<=73 else m-73}"
    print(num(m1), num(m2))
    r_grp_m1 = list_red_members(mols[m1-1])
    r_grp_m2 = list_red_members(mols[m2-1])
    for r_grp_1, r_grp_2 in product(r_grp_m1, r_grp_m2):        
        b2s = find_isomorphic(mols[m1-1], mols[m2-1], r_grp_1)
        if not b2s: continue
        b1s = find_isomorphic(mols[m2-1], mols[m1-1], r_grp_2)
        if not b1s: continue
        for b2 in b2s:
            for b1 in b1s:
                if check_isomorphic(mols[m1-1], mols[m2-1], r_grp_1+b1, b2+r_grp_2):
                    if connected(mols[m1-1], r_grp_1+b1) and connected(mols[m2-1], r_grp_2+b2):
                        print(inc(r_grp_1+b1), inc(b2+r_grp_2))
                        return True
    return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motifs_folder')
    parser.add_argument('--extra_label_path')
    args = parser.parse_args()
    mols = load_mols(args.motifs_folder)
    red_grps = annotate_extra(mols, args.extra_label_path)
    # sanity checks
    print(red_isomorphic(mols, 48, 73))
    os.makedirs(os.path.join(args.motifs_folder, f'with_label/'), exist_ok=True)
    for i, mol in enumerate(mols):
        Draw.MolToFile(mol, os.path.join(args.motifs_folder, f'with_label/{i+1}.png'), size=(2000, 2000))
