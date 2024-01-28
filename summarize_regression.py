from argparse import ArgumentParser
import os
import json
import pandas as pd
import numpy as np

parser = ArgumentParser()
parser.add_argument('--logs')
parser.add_argument('--rank_metric', default='avg_best_mae')
parser.add_argument('--report_metric', default='avg_best_r2')
args = parser.parse_args()
# sample usage: python summarize_regression.py --logs /home/msun415/polymer_walk/logs/logs-1704913899.3969326/

best_metrics = {}
best_metric_descrs = {}
for f in os.listdir(args.logs):
    folder = os.path.join(args.logs, f)
    if os.path.exists(os.path.join(folder, 'config.json')):
        try:
            config = json.loads(json.load(open(os.path.join(folder, 'config.json'))))
        except:
            config = json.load(open(os.path.join(folder, 'config.json')))
        if 'test_seed' not in config:
            continue
        if 'motifs_folder' in config:
            if 'hopv' in config['motifs_folder']:
                if 'norm_metrics' not in config or not config['norm_metrics']:
                    continue
        else:
            continue
        options = []
        for (option, val) in config.items():
            if val is False or val is None:
                continue
            if isinstance(val, list):
                val = ' '.join(map(str, val))
            if val is True:
                option_str = f'--{option}'
            else:
                option_str = f'--{option}'+f' {val}'
            options.append(option_str)
        command = 'python main.py ' + ' '.join(options)
        test_seed = config['test_seed']
        if test_seed == -1:
            continue        
        if os.path.exists(os.path.join(folder, 'metrics.csv')):
            try:
                metrics = pd.read_csv(os.path.join(folder, 'metrics.csv'), index_col=0)
                best_metric = metrics[args.rank_metric]                    
                best_epoch = best_metric.argmin()
            except:
                continue
            best_metric = best_metric.min()
            best_report_metric = metrics[args.report_metric][best_epoch]
            if test_seed not in best_metrics or best_metric < best_metrics[test_seed][args.rank_metric]:
                best_acc_descr = f"test seed: {test_seed}, best epoch: {best_epoch}, best mae: {best_metric}, best r2: {best_report_metric}"
                best_acc_descr += f"\nfolder: {folder}"
                best_acc_descr += f"\ncomamnd: {command}"
                best_metrics[test_seed] = {args.rank_metric: best_metric, args.report_metric: best_report_metric}
                best_metric_descrs[test_seed] = best_acc_descr
print('\n'.join(list(best_metric_descrs.values())))
for k in best_metrics[list(best_metrics)[0]]:
    k_metrics = []
    for metric_dict in best_metrics.values():
        k_metrics.append(metric_dict[k])
    print(f"{k} mean={np.mean(k_metrics)}, std={np.std(k_metrics)}")
