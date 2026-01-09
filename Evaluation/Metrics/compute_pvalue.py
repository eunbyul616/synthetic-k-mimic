import numpy as np
from sklearn.metrics import average_precision_score

from Evaluation.Metrics.compare_auc_delong_xu import *


def compute_auc_p_value(y_true, y_proba1, y_proba2):
    p_value = delong_roc_test(
        y_true,
        y_proba1,
        y_proba2
    )[0][0]

    return 10 ** p_value


def compute_auprc_p_value(auc1, auc2):
    # from scipy.stats import ttest_rel
    #
    # t_stat, p_value = ttest_rel(np.array(auc1), np.array(auc2))

    from scipy.stats import wilcoxon

    stat, p_value = wilcoxon(np.array(auc1), np.array(auc2))

    return p_value


def p_value_to_star(p):
    if p < 0.001:
        return "***", "p < 0.001"
    elif p < 0.01:
        return "**", "p < 0.01"
    elif p < 0.05:
        return "*", "p < 0.05"
    else:
        return "", ''
