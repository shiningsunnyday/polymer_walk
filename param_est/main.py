from models import MRFModel

import torch
from dataloader import TreeDataset
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import QED

import argparse


def main(args):
    train_dataset = TreeDataset(path="./dags/depth-2-train.json")
    
    mrf_model = MRFModel(mol_feat_dim=args.mol_feat_dim, 
                         hidden_size=args.hidden_size, 
                         bb_sample_size=args.bb_sample_size, 
                         rxt2bb_pair=train_dataset.get_rxt2bb_pairs())
    
    optimizer = optim.Adam(mrf_model.parameters(), lr=args.learning_rate)

    for epoch in args.max_epoches:
        sample_rxt = mrf_model.sample_rxt()
        
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCMC training')
    parser.add_argument('--mol_feat_dim', type=int, default=1024)
    parser.add_argument('--bb_sample_size', type=int, default=500)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--max_epoches', type=int, default=200)

    args = parser.parse_args()
    main(args)