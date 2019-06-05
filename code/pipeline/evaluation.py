import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.svm import LinearSVC
# taken in large part from rayid ghani's magicloops 

score_function_overrides = {
    LinearSVC: LinearSVC.decision_function
}

metrics = {
    "precision": precision_score,
    "recall"   : recall_score,
    "accuracy" : accuracy_score,
    "f1_score" : f1_score
    # auc_roc run on scores, not binarized predictions
} 

def apply_threshold(threshold, scores):
    return np.where(scores > threshold, 1, 0)

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_score, k):
    cutoff_index = int(len(y_score) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_score))]
    # return binarized predictions and score threshold
    return (predictions_binary, y_score[cutoff_index])

def evaluate(positive_label, k_values, y_true, y_score):
    y_score_sorted, y_true_sorted = joint_sort_descending(np.array(y_score), np.array(y_true))
    evaluation = {"roc-auc-score" : roc_auc_score(y_true, y_score)}
    for k in k_values:
        (preds_at_k, threshold) = generate_binary_at_k(y_score_sorted, k)
        evaluation["threshold-at-" + str(k)] = threshold
        evaluation["accuracy-at-"  + str(k)] = accuracy_score( y_true, preds_at_k)
        evaluation["precision-at-" + str(k)] = precision_score(y_true, preds_at_k, pos_label=positive_label)
        evaluation["recall-at-"    + str(k)] = recall_score(   y_true, preds_at_k, pos_label=positive_label)
        evaluation["f1-at-"        + str(k)] = f1_score(       y_true, preds_at_k, pos_label=positive_label)
    
    return (evaluation, precision_recall_curve(y_true, y_score, positive_label))

def find_best_model(model_evaluations, metric_type, k, num_results = 5):
    ''' idea: 
        concatenate argument to find column name
        return top 5 models in that split
        return the associated model name and the stat
    '''

    metric = metric_type + "-at-" + k

    return model_evaluations.groupby('name')[metric] \
                                .mean(metric) \
                                    .sort_values(metric, ascending = False) \
                                        .iloc[0:num_results-1]