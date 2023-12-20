from networkx.readwrite import json_graph
from rdkit import Chem
import torch
from torch.utils.data import Dataset

import json
import numpy as np


class TreeDataset():
    def __init__(self, path="./dags/depth-2-train.json"):
        with open(path) as f:
            data_all = json.load(f)
        
        nx_data_all = []
        for data in data_all:
            nx_data = json_graph.tree_graph(data)
            nx_data_all.append(nx_data)
        
        self.mol_feat_dim = 1024

        # get all building blocks, reactions, and products
        # make all trees to have two children, for single bb, use None as the second child (will be embedded as zero vector)
        self.all_building_blocks = {None: np.zeros(self.mol_feat_dim)}
        self.all_reactions = set()
        self.all_products = {}
        self.all_trees = []
        
        for nx_data in nx_data_all:
            # reaction are the node with "rxn_id"
            reaction_id_node = [x for x in nx_data.nodes() if "rxn_id" in nx_data.nodes[x]]
            assert len(reaction_id_node) == 1
            
            # building blocks are nodes with in_degree = 0
            building_blocks_node = [x for x in nx_data.nodes() if nx_data.in_degree(x) == 0 and x != reaction_id_node]
            assert len(building_blocks_node) == 2 or len(building_blocks_node) == 1
            
            # product is the node with out_degree = 1
            product_node = [x for x in nx_data.nodes() if nx_data.out_degree(x) == 1 and x != reaction_id_node]
            assert len(product_node) == 1
            
            assert set(building_blocks_node + product_node + reaction_id_node) == set(nx_data.nodes())
            reaction_id_node = reaction_id_node[0]
            product_node = product_node[0]
            
            building_blocks_sml = []
            for bb_n in building_blocks_node:
                bb_sml = nx_data.nodes[bb_n]["smiles"]
                building_blocks_sml.append(bb_sml)
                if bb_sml not in self.all_building_blocks.keys():
                    mol = Chem.MolFromSmiles(bb_sml)
                    fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.mol_feat_dim)
                    fp_array = np.array(fp, dtype=np.float)
                    self.all_building_blocks[bb_sml] = fp_array
                
            product_sml = nx_data.nodes[product_node]["smiles"]
            if product_sml not in self.all_products.keys():
                mol = Chem.MolFromSmiles(product_sml)
                fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.mol_feat_dim)
                fp_array = np.array(fp, dtype=np.float)
                self.all_products[product_sml] = fp_array
            
            reaction_id = nx_data.nodes[reaction_id_node]["rxn_id"]
            self.all_reactions.add(nx_data.nodes[product_node]["rxn_id"])
            
            tree = (None, None, reaction_id, product_sml)
            for bb_i, bb in enumerate(building_blocks_sml):
                tree[bb_i] = bb
                
            self.all_trees.append(tree)
            
        import pdb; pdb.set_trace()
    
    # def get_all_bb_as_tensor(self):
    #     all_bb = np.array(list(self.all_building_blocks.values()))
    #     return torch.from_numpy(all_bb)
    
    def get_rxt2bb_pairs(self):
        rxt2bb_pairs = {}
        for tree in self.all_trees:
            bb_0, bb_1, rxn, product = tree
            if product not in rxt2bb_pairs.keys():
                rxt2bb_pairs[product] = []
            
            bb_pair = np.concatenate([self.all_building_blocks[bb_0], self.all_building_blocks[bb_1]], axis=1)
            import pdb; pdb.set_trace()
            
            rxt2bb_pairs[product].append(bb_pair)
        
        for rxt in rxt2bb_pairs.keys():
            rxt2bb_pairs[rxt] = np.stack(rxt2bb_pairs[rxt])
            
        import pdb; pdb.set_trace()
        return rxt2bb_pairs
        
    def __len__(self):
        return len(self.all_trees)
