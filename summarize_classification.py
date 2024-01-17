from argparse import ArgumentParser
import os
import json
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--logs')
args = parser.parse_args()

best_accs = {}
best_acc_descrs = {}
for f in os.listdir(args.logs):
    folder = os.path.join(args.logs, f)
    if os.path.exists(os.path.join(folder, 'config.json')):
        config = json.loads(json.load(open(os.path.join(folder, 'config.json'))))
        if os.path.exists(os.path.join(folder, 'metrics.csv')):
            metrics = pd.read_csv(os.path.join(folder, 'metrics.csv'), index_col=0)
            test_seed = config['test_seed']
            best_epoch = metrics['avg_acc'].argmax()
            best_acc = metrics['avg_acc'].max()
            best_epoch_auc = metrics['avg_auc'][best_epoch]
            if test_seed == -1:
                continue
            if test_seed not in best_accs or best_acc > best_accs[test_seed]:
                best_acc_descr = f"test seed: {test_seed}, best epoch: {best_epoch}, best acc: {best_acc}, best epoch auc: {best_epoch_auc}"
                best_accs[test_seed] = best_acc
                best_acc_descrs[test_seed] = best_acc_descr
print('\n'.join(list(best_acc_descrs.values())))
