from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

import numpy as np

import math

def __calculate_metric(y_true, y_pred, metric):    
    if len(y_true.shape) == 1:
        score = metric(y_true, y_pred)
        return score, [score]

    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = metric(y_true[:,i], y_pred[:,i])
            if not math.isnan(score):
                scores.append(score)
            else:
                print('col', i, 'returned nan')
        except Exception as e:
            print(e)
    avg_score = np.mean(scores)
    return avg_score, np.array(scores)


def calculate_metric(y_true, y_pred, metric):
    metric = metric.lower()
    y_pred_bin = (y_pred > 0.5)
    #print('Errors for metric ', metric)

    if metric == 'auc':
        return __calculate_metric(y_true, y_pred, roc_auc_score)
    if metric == 'map':
        return __calculate_metric(y_true, y_pred, average_precision_score)
    if metric == 'precision':
        return __calculate_metric(y_true, y_pred_bin, precision_score)
    if metric == 'recall':
        return __calculate_metric(y_true, y_pred_bin, recall_score)
    if metric == 'f1':
        return __calculate_metric(y_true, y_pred_bin, f1_score)

    print('Metric {} not found. Please check spelling.'.format(metric))
    return np.empty(0), np.empty(0) 


def get_short_metrics_message(bin_auc, bin_f1, labels_auc, labels_map, labels_f1):

    ml_score = (labels_auc + labels_map) / 2.0
    model_score = (bin_auc + ml_score) / 2.0

    msg = 'ml_auc, ml_map, ml_f1, bin_auc, bin_f1, ml_score, model_score\n'
    msg += '{}\n{}\n{}\n{}\n{}\n{}\n{}'.format(labels_auc, labels_map, labels_f1, bin_auc, bin_f1, ml_score, model_score)

    return msg

def get_full_metrics_message(bin_auc, bin_f1, labels_auc, labels_map, labels_f1):
    ml_score = (labels_auc + labels_map) / 2.0
    model_score = (bin_auc + ml_score) / 2.0

    msg = '----- Multilabel scores -----\n'
    msg += 'auc_score: {}\n'.format(labels_auc)
    msg += 'mAP: {}\n'.format(labels_map)
    msg += 'f1: {}\n'.format(labels_f1)
    msg += '----- Binary scores -----\n'
    msg += 'auc: {}\n'.format(bin_auc)
    msg += 'f1: {}\n'.format(bin_f1)
    msg += '----- Multilabel Score -----\n'
    msg += 'ml_score: {}\n'.format(ml_score)
    msg += '----- Final Score -----\n'
    msg += 'model_score: {}\n'.format(model_score)
    msg += '----- Copy-Paste -----\n'
    msg += '{}\n{}\n{}\n{}\n{}\n{}\n{}'.format(labels_auc, labels_map, labels_f1, bin_auc, bin_f1, ml_score, model_score)

    return msg


def get_scores(y_true, y_pred, normal_col_idx):
    bin_auc, scores_auc = calculate_metric(y_true[:, normal_col_idx], y_pred[:, normal_col_idx], 'AUC')
    #bin_map, scores_map = calculate_metric(y_true[:, 0], y_pred[:, 0], 'mAP')
    bin_f1, scores_f1 = calculate_metric(y_true[:, normal_col_idx], y_pred[:, normal_col_idx], 'f1')

    y_true_labels = np.delete(y_true, normal_col_idx, axis=1)
    y_pred_labels = np.delete(y_pred, normal_col_idx, axis=1)
    
    labels_auc, scores_auc = calculate_metric(y_true_labels, y_pred_labels, 'AUC')
    labels_map, scores_map = calculate_metric(y_true_labels, y_pred_labels, 'mAP')
    labels_f1, scores_f1 = calculate_metric(y_true_labels, y_pred_labels, 'f1')

    scores_auc = np.concatenate(([bin_auc], scores_auc))
    scores_map = np.concatenate(([0], scores_map))
    scores_f1 = np.concatenate(([bin_f1], scores_f1))

    return np.array([bin_auc, bin_f1, labels_auc, labels_map, labels_f1]), scores_auc, scores_map, scores_f1  