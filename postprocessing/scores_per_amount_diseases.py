import pandas as pd
import numpy as np

from utils.metrics import get_scores_all


y_true_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\drop_all\\20_labels\\merged_20_labels_drop_10.0_perc.csv'
y_pred_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\codes\\trained_models\\merged_subsets\\swin\\preds.csv'
idxs_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\codes\\trained_models\\merged_subsets\\swin\\fold_0\\val_idx.csv'


y_true = pd.read_csv(y_true_path)
y_pred = pd.read_csv(y_pred_path, header=None)
idxs = pd.read_csv(idxs_path, header=None).iloc[:, 0].to_numpy()

y_true = y_true.iloc[idxs, 4:]

max_diseases = y_true.sum(axis=1).max()

group_idxs = (y_true.sum(axis=1) == 3).to_numpy()

#print(group_idxs)

print(y_true.iloc[group_idxs, :].sum())
print((y_pred.iloc[group_idxs, :]>0.5).sum())

print(get_scores_all(y_true.iloc[group_idxs].to_numpy(), y_pred.iloc[group_idxs].to_numpy()))
