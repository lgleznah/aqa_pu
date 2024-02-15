import numpy as np

# TODO: This metric returns values greater than 1. There is an error somewhere
def f1_pu(y_true, y_pred):
    '''
    F1-score estimate for PU problems

    This score is estimated for PU problems as the recall square over the ratio of positive predictions,
    according to "Learning with positive and unlabeled examples using weighted logistic regression", 
    by Lee, W.S. and Liu, B.

                     r^2
    F1(y_hat) = -------------
                 Pr(y_hat=1)
    '''
    pr_yhat_positive = len(y_pred[y_pred > 0.5]) / len(y_pred)
    recall = np.sum((y_true > 0.5) & (y_pred > 0.5)) / np.sum(y_true > 0.5)

    print(f'Recall: {recall}, probability_prediction_positive: {pr_yhat_positive}')
    print(f'True positives: {np.sum((y_true > 0.5) & (y_pred > 0.5))}, positives: {np.sum(y_true > 0.5)}')

    print(y_true > 0.5)
    print(y_pred > 0.5)

    return recall * recall / pr_yhat_positive

def aul_pu(y_true, y_pred):
    '''
    Area-under-lift curve for PU problems.

    As proven by Liwei Jiang et al. in "Improving Positive Unlabeled Learning: Practical AUL Estimation and New
    Training Method for Extremely Imbalanced Data Sets", this metric approaches its equivalent for traditional 
    binary classification problems when the number of samples is relatively large.

    Code implementation follows what is shown on "AUL IS A BETTER OPTIMIZATION METRIC IN PU LEARNING"
    '''
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    def s(a, b):
        if (a > b):
            return 1.0
        elif (a < b):
            return 0.0
        return 0.5
    
    n_labeled = np.sum(y_true == 1)
    n_total = len(y_true)
    factor = 1.0 / (n_labeled * n_total)
    acc = 0

    labeled_predictions = y_pred[y_true == 1]
    for labeled_pred in labeled_predictions:
        for pred in y_pred:
            acc += s(labeled_pred, pred)

    return factor * acc

def lift_curve_pu(y_true, y_pred):
    '''
    Returns a list of (x,y) points representing the PU Lift Curve for the given data.
    '''
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    preds_ordering = np.argsort(y_pred)
    y_pred_sorted = y_pred[preds_ordering]
    y_true_sorted = y_true[preds_ordering]
    n = len(y_pred_sorted)
    p_count = len(y_true_sorted[y_true_sorted == 1])

    x,y = [],[]

    for threshold in y_pred_sorted:
        y_pred_thresh = y_pred_sorted > threshold
        tp_count = np.sum((y_true_sorted == 1) & (y_pred_thresh == 1))
        fp_count = np.sum((y_true_sorted == 0) & (y_pred_thresh == 1))
        y_rate = (tp_count + fp_count) / n

        tpr = tp_count / p_count

        x.append(y_rate)
        y.append(tpr)

    return x,y