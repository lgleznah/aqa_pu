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

def aul_pu(y_true, y_pred, step=0.01):
    '''
    Area-under-lift curve for PU problems.

    As proven by Liwei Jiang et al. in "Improving Positive Unlabeled Learning: Practical AUL Estimation and New
    Training Method for Extremely Imbalanced Data Sets", this metric approaches its equivalent for traditional 
    binary classification problems when the number of samples is relatively large.

    Code for getting the lift curve gotten from:
    https://howtolearnmachinelearning.com/articles/the-lift-curve-in-machine-learning/
    '''

    # Sort predictions to get lift curve
    y_pred_sorted_idxs = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[y_pred_sorted_idxs]

    x_val = np.arange(step,1+step,step)
    ratio_ones = y_true.sum() / len(y_true)
    curve_values = []

    for x in x_val:
        num_data = int(np.ceil(x*len(y_pred)))
        data_here = y_true_sorted[:num_data]
        ratio_ones_here = data_here.sum() / len(data_here)
        curve_values.append(ratio_ones_here / ratio_ones)

    # Integrate lift curve to get AUL metric
    metric = 0.0
    for value in curve_values:
        metric += value * step

    return metric