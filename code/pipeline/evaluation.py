import numpy as np
from sklearn.metrics import precision_score, recall_score

# taken in large part from rayid ghani's magicloops 

def apply_threshold(threshold, scores):
    return np.where(scores > threshold, 1, 0)

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_score, k):
    cutoff_index = int(len(y_score) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_score))]
    return predictions_binary

def metrics_at_k(name, index, k, y_true, y_score):
    y_score_sorted, y_true_sorted = joint_sort_descending(np.array(y_score), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_score_sorted, k)

    report = classification_report(y_true, preds_at_k, output_dict=True)
    evaluation.update({
    metric + "-k" + str(k): value
    for (metric, value)
    in report[self.evaluation_key].items()})

    precision = precision_score(y_true_sorted, preds_at_k)
    recall = recall_score(y_true_sorted, preds_at_k)

    return { 
        "precision-at-k{}-t{}".format(k, t): precision,
        "recall-at-k{}-t{}".format(k, t) : recall
    }
