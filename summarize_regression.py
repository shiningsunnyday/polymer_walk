from argparse import ArgumentParser
import os
import json
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--logs')
args = parser.parse_args()

best_metrics = {}
best_metric_descrs = {}
for f in os.listdir(args.logs):
    folder = os.path.join(args.logs, f)
    if os.path.exists(os.path.join(folder, 'config.json')):
        config = json.loads(json.load(open(os.path.join(folder, 'config.json'))))
        test_seed = config['test_seed']
        if test_seed == -1:
            continue        
        if os.path.exists(os.path.join(folder, 'metrics.csv')):
            metrics = pd.read_csv(os.path.join(folder, 'metrics.csv'), index_col=0)
            best_metric = metrics['avg_best_mae']    
            best_epoch = best_metric.argmin()
            best_metric = best_metric.max()
            best_report_metric = metrics['avg_best_r2'][best_epoch]
            if test_seed not in best_metrics or best_metric < best_metrics[test_seed]:
                best_acc_descr = f"test seed: {test_seed}, best epoch: {best_epoch}, best mae: {best_metric}, best r2: {best_report_metric}"
                best_metrics[test_seed] = best_metric
                best_metric_descrs[test_seed] = best_acc_descr
print('\n'.join(list(best_metric_descrs.values())))
