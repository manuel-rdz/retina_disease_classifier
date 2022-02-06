import pandas as pd
import numpy as np

from utils.metrics import  get_scores, get_short_metrics_message

y_true_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\drop_all\\20_labels\\merged_20_labels_drop_10.0_perc.csv'
test_idxs = 'C:\\Users\\AI\\Desktop\\student_Manuel\\codes\\trained_models\\comparison\\20220119-212316-efficientnet_b3-630493\\fold_0\\val_idx.csv'
logits_paths = [
    'C:\\Users\\AI\\Desktop\\student_Manuel\\papers\\comparison\\RIADD_winner\\small_model\\preds_b5.csv',
    'C:\\Users\\AI\\Desktop\\student_Manuel\\papers\\comparison\\RIADD_winner\\small_model\\preds_b6.csv']


combined_logits = pd.DataFrame()

for path in logits_paths:
    current_logits = pd.read_csv(path, header=None, dtype=float)

    if combined_logits.empty:
        combined_logits = current_logits
    else:
        combined_logits += current_logits

combined_logits /= (len(logits_paths) * 1.0)

idxs = pd.read_csv(test_idxs, header=None).iloc[:, 0].to_numpy()
y_true = pd.read_csv(y_true_path).iloc[idxs, 4:].to_numpy(dtype=np.float32)
combined_logits = combined_logits.to_numpy(dtype=np.float32)

scores, _, _, _ = get_scores(y_true, combined_logits, 1)
print(get_short_metrics_message(*scores))

