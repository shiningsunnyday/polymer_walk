import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.data import Data, Dataset, DataLoader

from dataset.dataset_test import MolFeatExtractDataset
import json


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FeatExtractor(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.dataset = dataset

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def feat_extract(self):
        dataloader = DataLoader(
            self.dataset, batch_size=1, shuffle=False,
            num_workers=0, drop_last=False
        )

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        model.eval()
        os.makedirs(self.config['dataset']['save_path'], exist_ok=True)
        
        for bn, data in enumerate(dataloader):
            data = data.to(self.device)
            feat, _ = model(data)
            
            # save features
            np.save(os.path.join(self.config['dataset']['save_path'], "{}.npy".format(data.data_name_idx[0].item())), feat.squeeze().detach().cpu().numpy())
        

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model


def main(config):
    dataset = MolFeatExtractDataset(config['dataset']['data_path'], 'regression')
    fine_tune = FeatExtractor(dataset, config)
    fine_tune.feat_extract()
    

if __name__ == "__main__":
    config = yaml.load(open("configs/permeability_gin.yaml", "r"), Loader=yaml.FullLoader)
    config['dataset']['task'] = 'regression'
    config['dataset']['data_path'] = '/research/cbim/vast/zz500/Projects/mhg/ICML2024/polymer_walk/data/datasetA_permeability/all_groups/'
    config['dataset']['save_path'] = '/research/cbim/vast/zz500/Projects/mhg/ICML2024/polymer_walk/data/datasetA_permeability/gin_pretrained_feat/'
    
    config['dataset']['target'] = "log10_He_Bayesian" # dummy target
    
    print(config)
    result = main(config)
            
    # os.makedirs('experiments', exist_ok=True)
    # df = pd.DataFrame(results_list)
    # df.to_csv(
    #     'experiments/{}_{}_finetune.csv'.format(config['fine_tune_from'], config['task_name']), 
    #     mode='a', index=False, header=False
    # )