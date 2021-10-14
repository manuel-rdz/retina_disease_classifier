import pandas as pd
import numpy as np

import os

from utils.metrics import get_metrics_message, get_scores


y_true_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\clean_merged_dataset_all.csv'
y_pred_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\codes\\trained_models\\Merged_all\\20211011-111059-vit_base_patch16_384\preds.csv'
output_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\codes\\trained_models\\Merged_all\\20211011-111059-vit_base_patch16_384'

n_datasets = 3

y_true = pd.read_csv(y_true_path)
y_pred = pd.read_csv(y_pred_path, header=None)

output_path = os.path.join(output_path, 'per_dataset_scores')
os.makedirs(output_path, exist_ok=True)

original_cols = y_true.columns.values[n_datasets+1:]

score_comparison = np.zeros((len(original_cols), n_datasets * 2))
header_score_comparison = []
start_comparison_cols = 0

for col_idx in range(1, n_datasets + 1):
    name = y_true.columns.values[col_idx]
    idxs = y_true.iloc[:, col_idx] == 1

    idxs = pd.Series(idxs, name='bools')

    dataset_x = y_true.iloc[idxs.values, 0].to_numpy()

    dataset_y_true = y_true.iloc[idxs.values, n_datasets + 1:].to_numpy(dtype=np.float32)
    dataset_y_pred = y_pred.iloc[idxs.values, :].to_numpy(dtype=np.float32)

    valid_labels = np.sum(dataset_y_true, axis=0) > 0.
    dataset_y_true = dataset_y_true[:, valid_labels]
    dataset_y_pred = dataset_y_pred[:, valid_labels]

    cols = original_cols[valid_labels]

    x_col = y_true.columns.values[0]

    avg_metrics, auc_scores, map_scores = get_scores(dataset_y_true, dataset_y_pred)
    message = get_metrics_message(*avg_metrics)

    score_comparison[valid_labels, start_comparison_cols] = auc_scores
    score_comparison[valid_labels, start_comparison_cols + 1] = map_scores
    header_score_comparison.append(name + '_auc')
    header_score_comparison.append(name + '_map')
    start_comparison_cols += 2

    os.makedirs(os.path.join(output_path, name), exist_ok=True)

    np.savetxt(os.path.join(output_path, name, 'preds.csv'), 
        np.column_stack((dataset_x, dataset_y_pred)),
        header = ",".join(map(str, np.concatenate(([x_col], cols)))),
        delimiter =", ", 
        fmt ='% s',
        comments='')

    np.savetxt(os.path.join(output_path, name, 'scores.csv'),
        np.column_stack((cols, auc_scores, map_scores)),
        header='Label,AUC,mAP',
        delimiter=', ',
        fmt='% s',
        comments='')

    message = get_metrics_message(*avg_metrics)

    f = open(os.path.join(output_path, name, "final_scores.txt"), "w")
    f.write(message)
    f.close()
    
    print()
    print(name + ':')
    print(message)

header_score_comparison.insert(0, 'Label')

np.savetxt(os.path.join(output_path, 'scores_comparison.csv'),
    np.column_stack((original_cols, score_comparison)),
    header=",".join(map(str, header_score_comparison)),
    delimiter=', ',
    fmt='% s',
    comments='')