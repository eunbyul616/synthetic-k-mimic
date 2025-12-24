import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_curve, roc_curve, auc


def auprc(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc_score = auc(recall, precision)

    return auprc_score