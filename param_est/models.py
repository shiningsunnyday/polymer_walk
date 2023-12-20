import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PotentialNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PotentialNN, self).__init__()
        # Define layers
        self.mol_layer = nn.Linear(input_size, hidden_size)
        self.rxn_layer = nn.Embedding(1, hidden_size)
        self.layer1 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
    
    def forward(self, mol, rxn_id):
        mol_feat = F.relu(self.mol_layer(mol))
        rxn_feat = F.relu(self.rxn_layer(rxn_id))
        x = torch.cat((mol_feat, rxn_feat), dim=1)
        x = F.relu(self.layer1(x))
        return self.layer2(x)


class MRFModel(nn.Module):
    def __init__(self, mol_feat_dim, hidden_size, bb_sample_size, rxt2bb_pair):
        super(MRFModel, self).__init__()
        self.mol_feat_dim = mol_feat_dim
        self.hidden_size = hidden_size
        self.bb_sample_size = bb_sample_size
        self.rxt2bb_pair = rxt2bb_pair
        
        self.potential_net = PotentialNN(mol_feat_dim, hidden_size)
        self.sample_rxt_saved_log_prob = []
    
    def get_rxt_dist(self):
        # log(p(rxt)) = log(\sum_{bb_0} phi(rxt, bb_0)) + log(\sum_{bb_1} phi(rxt, bb_1))
        # Since it is hard to enumerate all the bbs, we do sampling
        # log(p(rxt)) = log(\sum_{i=1}^N{bb^i_0} phi(rxt, bb^i_0) / N) + log(\sum_{i=1}^N{bb^i_1} phi(rxt, bb^i_1) / N)
        
        log_p_rxt_all = []
        for rxt in self.rxt2bb_pair.keys():
            bb_pairs = self.rxt2bb_pair[rxt]
            
            num_bb_pairs, feat_dim = bb_pairs.shape
            assert feat_dim == self.mol_feat_dim * 2
            
            bb_pairs = bb_pairs[torch.randperm(num_bb_pairs)[:self.bb_sample_size]]
            
            bb_pairs_cut = torch.cat((bb_pairs[:, :self.mol_feat_dim], bb_pairs[:, self.mol_feat_dim:]), dim=0)
            import pdb; pdb.set_trace()
            
            rxt_tile = torch.tile(rxt, (self.bb_sample_size * 2, 1))
            import pdb; pdb.set_trace()
            
            potentials = self.potential_net(rxt_tile, bb_pairs_cut)
            import pdb; pdb.set_trace()
            
            potentials /= self.bb_sample_size
            import pdb; pdb.set_trace()
            
            potentials_0 = potentials[:self.bb_sample_size]
            potentials_1 = potentials[self.bb_sample_size:]
            
            log_p_rxt = torch.logsumexp(potentials_0, dim=0) + torch.logsumexp(potentials_1, dim=0)
            import pdb; pdb.set_trace()
            
            log_p_rxt_all.append(log_p_rxt)
        
        import pdb; pdb.set_trace()
        log_p_rxt_all = torch.stack(log_p_rxt_all)
        import pdb; pdb.set_trace()
        
        p_rxt_all = torch.exp(log_p_rxt_all)
        return p_rxt_all
    
    def sample_rxt(self, p_rxt_all):
        m = Categorical(p_rxt_all)
        a = m.sample()
        import pdb; pdb.set_trace()
        
        sample_rxt = list(self.rxt2bb_pair.keys())[a.item()]
        import pdb; pdb.set_trace()
        
        self.sample_rxt_saved_log_prob.append(m.log_prob(a))
        import pdb; pdb.set_trace()
        return sample_rxt

    def forward(self):
        p_rxt_all = self.get_rxt_dist()
        sample_rxt = self.sample_rxt(p_rxt_all)
        
        return sample_rxt