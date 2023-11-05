from unimol_tools import MyMolTrain, YamlHandler #, MolPredict
import os
import numpy as np
import json

data = "/research/cbim/vast/zz500/Projects/mhg/ICML2024/polymer_walk/datasets/datasetA_permeability.csv"
targets_str = [ "log10_CO2_Bayesian", "log10_H2_Bayesian", "log10_He_Bayesian", "log10_N2_Bayesian", "log10_O2_Bayesian", "log10_CH4_Bayesian" ]

results_dict = {}
num_expr = 5
for expr_id in range(num_expr):
    for tgt_str in targets_str:
        print("=============")
        
        config = os.path.join(os.path.dirname(__file__), 'finetune_configs/permeability.yaml')
        yamlhandler = YamlHandler(config)
        config = yamlhandler.read_yaml()
        
        config.target_col_prefix = tgt_str
        
        clf = MyMolTrain(config = config,
                        yamlhandler = yamlhandler,
                        task='regression', 
                        data_type='molecule', 
                        epochs=200, 
                        batch_size=16, 
                        metrics=['mae', 'r2'],
                        )

        pred = clf.fit(data = data)
        res = clf.model.cv['test_metric']
        
        if tgt_str not in results_dict.keys():
            results_dict[tgt_str] = {"metrics": [(res['mae'].item(), res['r2'].item())]}
        else:
            results_dict[tgt_str]["metrics"].append((res['mae'].item(), res['r2'].item()))

for target in results_dict.keys():
    all_mae = [r[0] for r in results_dict[target]["metrics"]]
    all_r2 = [r[1] for r in results_dict[target]["metrics"]]
    assert len(all_mae) == num_expr
    assert len(all_r2) == num_expr
    results_dict[target]['mae_mean'] = np.mean(all_mae).item()
    results_dict[target]['mae_std'] = np.std(all_mae).item()
    results_dict[target]['r2_mean'] = np.mean(all_r2).item()
    results_dict[target]['r2_std'] = np.std(all_r2).item()
    
with open('./unimol_finetune.json', 'w') as fp:
    json.dump(results_dict, fp)
        