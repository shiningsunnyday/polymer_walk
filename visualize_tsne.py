import argparse
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

colors = 'bgrcmykw'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('--gnn_feats')
    parser.add_argument('--test_feat_path')
    parser.add_argument('--data_file')
    parser.add_argument('--algo', default='tsne', choices=['tsne','pca'])
    parser.add_argument('--num_quantiles', type=int, default=2)
    
    args = parser.parse_args()

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

    assert len(os.listdir(args.gnn_feats)) == len(lines)
    gnn_feats = [None for _ in test_idxes]
    for f in os.listdir(args.gnn_feats):
        idx = int(f.split('.')[0])
        if idx in test_idxes:
            gnn_feats[test_idxes.index(idx)] = np.load(os.path.join(args.gnn_feats, f))
    gnn_feats = np.array(gnn_feats)
    test_feats = torch.load(test_feat_path) 
    if args.algo == 'tsne':
        tsne = TSNE(n_components=2, verbose=1)    
        X = tsne.fit_transform(test_feats)   
        X_gnn = tsne.fit_transform(gnn_feats)      
    else:
        pca = PCA(n_components=2)
        X = pca.fit_transform(test_feats)
        X_gnn = pca.fit_transform(gnn_feats)

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
    fig = plt.Figure(figsize=(10,10))
    ax = fig.add_subplot(1,2,1)
    ax.set_title("Our method's final layer features")
    ax_gnn = fig.add_subplot(1,2,2)
    ax_gnn.set_title("Pre-trained GIN final layer features")
    for i, mask in enumerate(masks):
        for idx, m in enumerate(mask):
            if m:
                ax.scatter(X[idx,0], X[idx,1], c=str(norm_scores[idx]), s=20)
                ax_gnn.scatter(X_gnn[idx,0], X_gnn[idx,1], c=str(norm_scores[idx]), s=20)
    fig.savefig(os.path.join(args.log_dir, 'tsne.png'))
    print(os.path.join(args.log_dir, 'tsne.png'))
