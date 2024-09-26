from argparse import ArgumentParser
import os
import json
import pandas as pd
import numpy as np
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from utils.data import plot_group_contrib, plot_group_contrib_with_err
import pickle
from collections import Counter

parser = ArgumentParser()
parser.add_argument('--logs')
parser.add_argument('--rank_metric', default='avg_best_mae')
parser.add_argument('--report_metric', default='avg_best_r2')
parser.add_argument('--file_to_screen', type=str)
args = parser.parse_args()
# sample usage: python summarize_regression.py --logs /home/msun415/polymer_walk/logs/logs-1704913899.3969326/


def summarize(args):
    best_metrics = {}
    best_metric_descrs = {}
    best_metric_commands = {}
    best_metric_configs = {}
    ignore_options = ['TORCH_SEED', 'input_dim', 'logs_folder']
    for f in os.listdir(args.logs):
        folder = os.path.join(args.logs, f)
        if os.path.exists(os.path.join(folder, 'config.json')):
            try:
                config = json.loads(json.load(open(os.path.join(folder, 'config.json'))))
            except:
                config = json.load(open(os.path.join(folder, 'config.json')))
            if 'test_seed' not in config:
                continue
            options = []
            for (option, val) in config.items():            
                if option in ignore_options:
                    continue
                if (val != 'share_params' and val is False) or val is None:
                    continue
                if isinstance(val, list):
                    val = ' '.join(map(str, val))
                if (option != 'share_params' and val is True) or (option == 'share_params' and val == False):
                    option_str = f'--{option}'
                else:
                    option_str = f'--{option}'+f' {val}'
                options.append(option_str)
            command = 'python main.py ' + ' '.join(options)
            test_seed = config['test_seed']
            if test_seed == -1:
                continue   
            if 'norm_metrics' not in config or not config['norm_metrics']:
                continue      
            if 'train_size' not in config or float(config['train_size']) != 1.0:
                continue
            if 'concat_mol_feats' not in config or config['concat_mol_feats']:
                continue

            # if 'edge_weights' not in config or not config['edge_weights']: # edge weights
            #     continue
            # if 'edge_weights' in config and config['edge_weights']: # no edge weights
            #     continue
            # if 'ablate_bidir' not in config or not config['ablate_bidir']: # ablate
            #     continue     
            if 'ablate_bidir' in config and config['ablate_bidir']: # don't ablate
                continue 
            
            if os.path.exists(os.path.join(folder, 'metrics.csv')):
                try:
                    metrics = pd.read_csv(os.path.join(folder, 'metrics.csv'), index_col=0)
                except:
                    continue
                if args.rank_metric not in metrics:
                    continue            
                best_metric = metrics[args.rank_metric]    
                if len(best_metric) == 0:
                    continue
                best_epoch = best_metric.argmin()   
                best_metric = best_metric.min()
                best_report_metric = metrics[args.report_metric][best_epoch]
                if test_seed not in best_metrics or best_metric < best_metrics[test_seed][args.rank_metric]:
                    best_acc_descr = f"test seed: {test_seed}, best epoch: {best_epoch}, best mae: {best_metric}, best r2: {best_report_metric}"
                    # best_acc_descr += f"\nfolder: {folder}"
                    command = command.replace('/home/msun415/polymer_walk/data/datasets/group-contrib/', '$dataset')
                    # command = command.replace('data/polymer_walks_v2_preprocess.txt', '$walksfile')
                    # command = command.replace('logs/logs-1703187675.0302503', '$logdir')
                    best_acc_descr += f"\ncomamnd: {command}"
                    if args.file_to_screen:
                        grammar_folder = config['grammar_folder']
                        predictor_file = find_ckpt(config['logs_folder'])
                        command = command.replace("red_graph_new_s21.edgelist", "red_graph.edgelist")
                        command += f' --grammar_file {grammar_folder}/ckpt.pt --predictor_file {predictor_file} --test_walks_file /home/msun415/polymer_walk/data/polymer_all_walks_1000.pkl'
                    best_metric_commands[test_seed] = command
                    best_metrics[test_seed] = {args.rank_metric: best_metric, args.report_metric: best_report_metric}
                    best_metric_descrs[test_seed] = best_acc_descr
                    best_metric_configs[test_seed] = config
    print('\n'.join(list(best_metric_descrs.values())))
    for k in best_metrics[list(best_metrics)[0]]:
        k_metrics = []
        for metric_dict in best_metrics.values():
            k_metrics.append(metric_dict[k])
        print(f"{k} mean={np.mean(k_metrics)}, std={np.std(k_metrics)}")    
    return best_metric_commands, best_metric_configs


def run_command(cmd):
    """ Function to execute a command in a subprocess """
    result = subprocess.run(cmd, text=True, capture_output=True, shell=True)
    return result.stdout, result.stderr


def find_ckpt(folder):
    res = None
    best = float('inf')
    for file in list(filter(lambda fi: '.pt' in fi, os.listdir(folder))):
        grps = re.match("predictor_ckpt_(.*).pt", file)
        if float(grps.groups()[0]) < best:
            best = float(grps.groups()[0])
            res = file
    if res is None:
        breakpoint()
    return os.path.join(folder, res)


def dfs(dag): 
    if dag.children:
        return [dag.val] + sum([dfs(c[0]) for c in dag.children], [])
    else:
        return [dag.val]    



def dfs_edge(dag):
    if dag.children:
        return [(dag.val, c[0].val) for c in dag.children]
    else:
        return []



if args.file_to_screen:
    data = pickle.load(open(args.file_to_screen, 'rb'))
    all_smiles = [d[1].smiles for d in data]
    all_dags = [d[1] for d in data]
    all_motifs = [dfs(dag) for dag in all_dags]    
    all_transitions = [dfs_edge(dag) for dag in all_dags]    
    paths = []
    all_commands = []
    err_threshs = {('H2','H2_N2'):[500,10],
                  ('O2','O2_N2'):[100,2],
                  ('CO2','CO2_CH4'):[1000,5]}
    for property, secondary_property, k, n in zip(['H2', 'O2', 'CO2'], ['H2_N2', 'O2_N2', 'CO2_CH4'], [9.765*1e4, 1.396*1e6, 5.369*1e6], [-1.4841, -5.666, -2.636]):
        all_outs = []
        all_out_2s = []
        all_orig_preds = []
        all_props = []
        if '_' in property:
            setattr(args, 'rank_metric', f'selectivity_{property}_mae')
            setattr(args, 'report_metric', f'selectivity_{property}_r^2')
        else:
            setattr(args, 'rank_metric', f'permeability_{property}_mae')
            setattr(args, 'report_metric', f'permeability_{property}_r^2')        

        best_metric_commands, best_metric_configs = summarize(args)
        conda_init_script = "/home/msun415/miniconda3/etc/profile.d/conda.sh"
        assert '.pkl' in args.file_to_screen    
        for test_seed in best_metric_commands:
            config = best_metric_configs[test_seed]
            predictor_dir = config['logs_folder']
            res = [os.path.join(predictor_dir, f"{prefix}.npy") for prefix in ['out', 'out_2', 'orig_preds', 'props']]
            if not np.all([os.path.exists(r) for r in res]):                
                command = best_metric_commands[test_seed]        
                command = f". {conda_init_script} && conda activate polymer_walk && export dataset=/home/msun415/polymer_walk/data/datasets/group-contrib/ && {command}"
                all_commands.append(command)                
            else:
                out, out_2, orig_preds, props = [np.load(file) for file in res]   
                all_outs.append(out)        
                all_out_2s.append(out_2)
                all_orig_preds.append(orig_preds)                
                all_props.append(props)
        col_names = [property, secondary_property]
        # path = plot_group_contrib(predictor_dir, out, out_2, orig_preds, props, col_names)                
        path, mask = plot_group_contrib_with_err(predictor_dir, all_outs, all_out_2s, all_orig_preds, all_props, col_names, err_thresh=err_threshs[(property, secondary_property)], k=k, n=n)        
        mask_smiles = np.array(all_smiles)[mask]
        mask_dags = [motif for motif, mask_ in zip(all_motifs, mask) if mask_]
        mask_transitions = [transition for transition, mask_ in zip(all_transitions, mask) if mask_]        
        mask_out_path = str(Path(path).parent / Path(path).stem) + '.txt'
        motif_out_path = str(Path(path).parent / Path(path).stem) + '_motifs.txt'
        transition_out_path = str(Path(path).parent / Path(path).stem) + '_transitions.txt'
        with open(mask_out_path, 'w+') as f:
            for smi in mask_smiles:
                f.write(f"{smi}\n")            
        mask_motifs = sum(mask_dags, [])
        mask_transitions = sum(mask_transitions, [])
        ct = Counter(mask_motifs)        
        with open(motif_out_path, 'w+') as f:
            for motif, count in ct.most_common():
                f.write(f"{motif} {count}\n")  
        ct = Counter(mask_transitions)    
        with open(transition_out_path, 'w+') as f:
            for transition, count in ct.most_common():
                f.write(f"{transition} {count}\n")                               
        print(mask_out_path)
        paths.append(path)        
    
    if len(all_commands):
        with ProcessPoolExecutor(max_workers=len(all_commands)) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(run_command, cmd) for cmd in all_commands]        
            # Retrieve results as they complete
            for future in futures:
                stdout, stderr = future.result()
                print("stdout:", stdout)
                print("stderr:", stderr)    

    scp_command = ""
    for i, path in enumerate(paths):
        if scp_command:
            prefix = "; "
        else:
            prefix = ""
        scp_command += prefix + f"scp msun415@f12.csail.mit.edu:/home/msun415/polymer_walk/{path} ~/Downloads/pareto_{i}.pdf"
        print(scp_command)
else:
    summarize(args)
