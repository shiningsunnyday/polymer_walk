import argparse
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40, 'font.family': 'serif'})
from matplotlib.colors import BoundaryNorm
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import os
import rdkit.Chem as Chem

colors = 'bgrcmykw'

def main(args):
    lines = open(args.data_file).readlines()
    test_idx_path = os.path.join(args.log_dir, 'test_idx.txt')
    test_idxes = list(map(int, open(test_idx_path).readlines()))
    best_loss = float("inf")

    if args.test_feat_path:
        test_feat_path = args.test_feat_path
    else:
        for f in os.listdir(args.log_dir):
            if 'test_feats' in f:
                score = f.split('_')[-1][:-3]
                if float(score) < best_loss:
                    best_loss = float(score)
                    test_feat_path = os.path.join(args.log_dir, f)
    test_feats = torch.load(test_feat_path) 

    if args.heuristic_feats:
        heuristic_feats = torch.load(args.heuristic_feats)     

    if os.path.isdir(args.gnn_feats):
        assert len(os.listdir(args.gnn_feats)) == len(lines)
        gnn_feats = [None for _ in test_idxes]
        for f in os.listdir(args.gnn_feats):
            idx = int(f.split('.')[0])
            if idx in test_idxes:
                gnn_feats[test_idxes.index(idx)] = np.load(os.path.join(args.gnn_feats, f))
        gnn_feats = np.array(gnn_feats)
    else:
        raise NotImplementedError
    if os.path.isfile(args.hmgnn_feats):
        assert '.npy' == args.hmgnn_feats[-4:]
        hmgnn_feats = np.load(args.hmgnn_feats)
        hmgnn_feats = hmgnn_feats[test_idxes]
    else:
        raise NotImplementedError

    test_feats = torch.load(test_feat_path)     
    if args.algo == 'tsne':
        tsne = TSNE(n_components=2, verbose=1, random_state=0, init='pca')    
        X = tsne.fit_transform(test_feats)   
        X_heuristic = tsne.fit_transform(heuristic_feats)   
        X_gnn = tsne.fit_transform(gnn_feats)      
        X_hmgnn = tsne.fit_transform(hmgnn_feats)      
    elif args.algo == 'pca':
        pca = PCA(n_components=2, random_state=0)
        X = pca.fit_transform(test_feats)
        X_heuristic = pca.fit_transform(heuristic_feats)
        X_gnn = pca.fit_transform(gnn_feats)
        X_hmgnn = pca.fit_transform(hmgnn_feats)
    else:
        N = test_feats.shape[0]
        dists = np.zeros((N, N))        
        # apply an ordering mask
        smis = [lines[idx].split(',')[0] for idx in test_idxes]
        sizes = [float(lines[idx].split(',')[1]) for idx in test_idxes]
        lines = [lines[idx] for idx in test_idxes]
        order = np.argsort(sizes)
        test_feats = test_feats[order]
        lines = [lines[idx] for idx in order]

        test_feats_norm = F.normalize(test_feats,dim=-1)

        boundaries = np.array([0, 0.001, 0.01, 0.1, 2])  # Adjust these values as needed for your data
        norm = BoundaryNorm(boundaries, ncolors=256, clip=True)
        # test_feats_norm = test_feats
        quantiles = [[] for _ in range(len(boundaries)-1)]
        for i in range(N):
            for j in range(N):
                dists[i, j] = cosine(test_feats_norm[i], test_feats_norm[j])
                # add to quantile
                if i <= j:
                    continue
                index = np.argmin(dists[i, j] >= boundaries)
                quantiles[index-1].append((i, j, lines[i], lines[j]))
        path = os.path.join(args.log_dir, 'dists.txt')
        f = open(path, 'w+')
        for i, quant in enumerate(quantiles):
            f.write(f"QUANTILE {i+1}: {boundaries[i]}-{boundaries[i+1]}\n")
            for idx1, idx2, smi1, smi2 in quant:
                smi1 = ' '.join(smi1.rstrip('\n').split(','))
                smi2 = ' '.join(smi2.rstrip('\n').split(','))
                f.write(f"{smi1} ({idx1})\n{smi2} ({idx2})\n\n")        


        fig = plt.Figure(figsize=(20,20))
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(dists, cmap='gray', norm=norm, interpolation='nearest')
        fig.colorbar(im, shrink=0.8)
        ax.set_title("Pairwise Cosine Distance on HOPV Test Set", fontsize=40, fontname='serif')
        ticks = range(0, N, 5)
        labels = [f"{tick}" if tick % 10 == 0 else '' for tick in ticks]
        ax.set_xticks(ticks=ticks, labels=labels, fontsize=30)
        ax.set_yticks(ticks=ticks, labels=labels, fontsize=30)
        path = os.path.join(args.log_dir, 'dists.png')
        fig.savefig(path)     
        return   


    data = []
    masks = [[] for _ in range(args.num_quantiles)]

    for idx in test_idxes:
        smi, *props = lines[idx].split(',')
        props = list(map(float, props))
        data.append([smi]+props)

    scores = np.array([x[1] for x in data])    
    sorted_scores = sorted(scores)
    quantile_scores = []
    for i in range(args.num_quantiles):
        quantile_scores.append(sorted_scores[int(i*len(sorted_scores)/args.num_quantiles)])
    for idx, (smi, *props) in zip(test_idxes, data):
        for i in range(args.num_quantiles):
            in_quantile = props[0]>=quantile_scores[i]
            if i < args.num_quantiles-1:
                in_quantile &= props[0]<quantile_scores[i+1]
            masks[i].append(in_quantile)

    min_score = min(scores)
    max_score = max(scores)
    norm_scores = [(s-min_score)/(max_score-min_score) for s in scores]
    fig = plt.Figure(figsize=(20,10))
    fig.suptitle('Final layer representations across methods', fontsize=36)    
    ax = fig.add_subplot(1,4,1)    
    ax.set_title("Our method")
    ax_heuristic = fig.add_subplot(1,4,2)
    ax_heuristic.set_title("Our method (-expert)")    
    ax_gnn = fig.add_subplot(1,4,3)
    ax_gnn.set_title("Pre-trained GIN")
    ax_hmgnn = fig.add_subplot(1,4,4)
    ax_hmgnn.set_title("HM-GNN")    
    f = open(os.path.join(args.log_dir, 'tsne.txt'),'w+')
    f.write(f"x,y,smiles,property\n")
    pos_ours, pos_heuristic, pos_gin, pos_hmgnn = [], [], [], []
    for i, mask in enumerate(masks):
        for idx, m in enumerate(mask):
            if m:
                ax.scatter(X[idx,0], X[idx,1], c=str(norm_scores[idx]), s=20)
                ax_heuristic.scatter(X_heuristic[idx,0], X_heuristic[idx,1], c=str(norm_scores[idx]), s=20)
                ax_gnn.scatter(X_gnn[idx,0], X_gnn[idx,1], c=str(norm_scores[idx]), s=20)
                ax_hmgnn.scatter(X_hmgnn[idx,0], X_hmgnn[idx,1], c=str(norm_scores[idx]), s=20)
                pos_ours.append((X[idx,0],X[idx,1],data[idx][0],data[idx][1]))
                pos_heuristic.append((X_heuristic[idx,0],X_heuristic[idx,1],data[idx][0],data[idx][1]))
                pos_gin.append((X_gnn[idx,0],X_gnn[idx,1],data[idx][0],data[idx][1]))
                pos_hmgnn.append((X_hmgnn[idx,0],X_hmgnn[idx,1],data[idx][0],data[idx][1]))
    pos_ours = sorted(pos_ours, key=lambda x:x[:2])
    pos_heuristic = sorted(pos_heuristic, key=lambda x:x[:2])
    f.write("=====Ours======\n")
    for p in pos_ours:
        f.write(",".join(map(str,p))+'\n')
    f.write("=====Ours (-expert)======\n")
    for p in pos_heuristic:
        f.write(",".join(map(str,p))+'\n')        
    f.write("=====GIN======\n")
    for p in pos_gin:
        f.write(",".join(map(str,p))+'\n')    
    f.write("=====HM-GNN======\n")
    for p in pos_hmgnn:
        f.write(",".join(map(str,p))+'\n')                    
    f.close()
    fig.savefig(os.path.join(args.log_dir, 'tsne.png'))    
    print(os.path.join(args.log_dir, 'tsne.png'))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('--gnn_feats')
    parser.add_argument('--heuristic_feats')
    parser.add_argument('--hmgnn_feats')
    parser.add_argument('--test_feat_path')
    parser.add_argument('--data_file')
    parser.add_argument('--algo', default='tsne', choices=['tsne','pca','dist'])
    parser.add_argument('--num_quantiles', type=int, default=2)
    
    args = parser.parse_args()

    main(args)
