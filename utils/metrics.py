from sklearn.metrics import roc_auc_score, average_precision_score

import numpy as np

import math

def auc_score(y_true, y_pred):
    if len(y_true.shape) == 1:
        score = roc_auc_score(y_true, y_pred)
        return score, [score]

    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = roc_auc_score(y_true[:,i], y_pred[:,i])
            if not math.isnan(score):
                scores.append(score)
        except ValueError:
            pass
    avg_score = np.mean(scores)
    return avg_score, np.array(scores)


def mAP_score(y_true, y_pred):
    if len(y_true.shape) == 1:
        score = average_precision_score(y_true, y_pred)
        return score, [score]
    
    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = average_precision_score(y_true[:, i], y_pred[:, i])
            if not math.isnan(score):
                scores.append(score)
        except ValueError:
            pass
    avg_score = np.mean(scores)
    return avg_score, np.array(scores)


def get_metrics_message(bin_auc, bin_map, labels_auc, labels_map):
    task2_score = (labels_auc + labels_map) / 2

    final_score = (bin_auc + task2_score) / 2

    msg = '----- Multilabel scores -----\n'
    msg += 'auc_score: {}\n'.format(labels_auc)
    msg += 'mAP: {}\n'.format(labels_map)
    msg += 'task score: {}\n'.format(task2_score)
    msg += '----- Binary scores -----\n'
    msg += 'auc: {}\n'.format(bin_auc)
    msg += 'mAP: {}\n'.format(bin_map)
    msg += '----- Final Score -----\n'
    msg += str(final_score)

    return msg


def get_scores(y_true, y_pred):
    bin_auc, scores_auc = auc_score(y_true[:, 0], y_pred[:, 0])
    bin_map, scores_map = mAP_score(y_true[:, 0], y_pred[:, 0])
    
    labels_auc, scores_auc = auc_score(y_true[:, 1:], y_pred[:, 1:])
    labels_map, scores_map = mAP_score(y_true[:, 1:], y_pred[:, 1:])

    scores_auc = np.concatenate(([bin_auc], scores_auc))
    scores_map = np.concatenate(([bin_map], scores_map))

    return np.array([bin_auc, bin_map, labels_auc, labels_map]), scores_auc, scores_map  