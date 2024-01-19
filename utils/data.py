import numpy as np 
from sklearn.metrics import r2_score


def pareto_or_not(prop_1, prop_2, num_pred, min_better=True):
    pareto_1 = []
    not_pareto_1 = []
    pareto_2 = []
    not_pareto_2 = []
    for i, (v_1, v_2) in enumerate(zip(prop_1, prop_2)):
        if min_better and v_2 <= prop_2[np.argwhere(prop_1 <= v_1)].min():
            if i < num_pred: pareto_1.append(i)
            else: pareto_2.append(i-num_pred)
        elif not min_better and v_2 >= prop_2[np.argwhere(prop_1 >= v_1)].max():
            if i < num_pred: pareto_1.append(i)
            else: pareto_2.append(i-num_pred)            
        else:
            if i < num_pred: not_pareto_1.append(i)
            else: not_pareto_2.append(i-num_pred)
    return pareto_1, not_pareto_1, pareto_2, not_pareto_2

