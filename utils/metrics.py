from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def auc_score(y_true, y_pred):
    if len(y_true.shape) == 1:
        score = roc_auc_score(y_true, y_pred)
        return score, [score]

    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = roc_auc_score(y_true[:,i], y_pred[:,i])
            scores.append(score)
        except ValueError:
            pass
    avg_score = np.mean(scores)
    return avg_score, scores


def mAP_score(y_true, y_pred):
    if len(y_true.shape) == 1:
        score = average_precision_score(y_true, y_pred)
        return score, [score]
    
    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = average_precision_score(y_true[:, i], y_pred[:, i])
            scores.append(score)
        except ValueError:
            pass
    avg_score = np.mean(scores)
    return avg_score, scores