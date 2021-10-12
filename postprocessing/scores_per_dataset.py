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

for col in range(1, n_datasets + 1):
    name = y_true.columns.values[col]
    idxs = y_true.iloc[:, col] == 1

    idxs = pd.Series(idxs, name='bools')

    dataset_y_true = y_true.iloc[idxs.values, n_datasets + 1:].to_numpy(dtype=np.float32)
    dataset_y_pred = y_pred.iloc[idxs.values, :].to_numpy(dtype=np.float32)

    print(dataset_y_true.shape)
    print(dataset_y_pred.shape)

    avg_metrics, auc_scores, map_scores = get_scores(dataset_y_true, dataset_y_pred)
    message = get_metrics_message(*avg_metrics)

    print(auc_scores.shape)
    print(map_scores.shape)

    os.makedirs(os.path.join(output_path, name), exist_ok=True)

    np.savetxt(os.path.join(output_path, name, 'preds.csv'), 
        dataset_y_pred,
        delimiter =", ", 
        fmt ='% s')

    np.savetxt(os.path.join(output_path, name, 'scores.csv'),
        np.column_stack((np.array(auc_scores), np.array(map_scores))),
        header='auc, map',
        delimiter=', ',
        fmt='% s')

    message = get_metrics_message(*avg_metrics)

    f = open(os.path.join(output_path, name, "final_scores.txt"), "w")
    f.write(message)
    f.close()

    print(message)


