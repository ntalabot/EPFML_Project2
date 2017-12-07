"""
Some metrics used to asses how precise is our fit:
    accuracy
    precision
    recall
    false positive rate
    false negative rate
    F1 score (https://en.wikipedia.org/wiki/F1_score)
    
https://en.wikipedia.org/wiki/Sensitivity_and_specificity
"""

import numpy as np

def true_false_pos_neg(y, y_pred):
    # Computes true/false positives/negatives
    TP = sum(int(y * y_pred == 1)) # The number of correctly predicted 1 labels
    TN = sum(int(y + y_pred == 0)) # The number of correctly predicted 0 labels
    FP = sum(int((y == 0) && (y_pred == 1))) # The number of incorrectly predicted 1 labels
    FN = sum(int((y == 1) && (y_pred == 0))) # The number of incorrectly predicted 1 labels
    
    return TP, TN, FP, FN

def accuracy(y, y_pred):
    TP, TN, FP, FN = true_false_pos_neg(y, y_pred)
    return (TP+TN)/(TP+TN+FP+FN)

def precision(y, y_pred):
    TP, _, FP, _ = true_false_pos_neg(y, y_pred)
    return TP/(TP+FP)

def recall(y, y_pred):
    TP, _, _, FN = true_false_pos_neg(y, y_pred)
    return TP/(TP+FN)

def false_pos_rate(y, y_pred):
    _, TN, FP, _ = true_false_pos_neg(y, y_pred)
    return FP/(TN+FP)

def false_neg_rate(y, y_pred):
    TP, _, _, FN = true_false_pos_neg(y, y_pred)
    return FN/(TP+FN)

def f1_score(y, y_pred):
    TP, _, FP, FN = true_false_pos_neg(y, y_pred)
    return 2*TP/(2*TP+FP+FN)