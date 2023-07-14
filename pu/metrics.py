import numpy as np

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
    recall = np.sum((y_true > 0.5) == (y_pred > 0.5)) / np.sum(y_true > 0.5)

    return recall * recall / pr_yhat_positive