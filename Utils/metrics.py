import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_ks_statistic(data1: pd.Series or np.array,
                           data2: pd.Series or np.array):
    """
    Calculate Kolmogorov-Smirnov statistics.
    Args:
        data1: Input data1
        data2: Input data2

    Returns:
        tuple: result of ks_2samp
    """
    res = ks_2samp(data1, data2)
    return res


def calculate_ks_statistics_by_feature(data1: pd.DataFrame,
                                       data2: pd.DataFrame,
                                       decimal: int=3):
    """
    Calculate Kolmogorov-Smirnov statistics by feature.
    Args:
        data1: Input data1
        data2: Input data2
        decimal: decimal places to round the result

    Returns:
        pd.DataFrame: KS statistics

    """
    ks_stats = []
    for col in data1.columns:
        # ignore missing values
        feature1 = data1[data1[col].notnull()][col]
        feature2 = data2[data2[col].notnull()][col]

        res = calculate_ks_statistic(feature1, feature2)
        ks_stat = np.round(res.statistic, decimal)
        p_value = np.round(res.pvalue, decimal)
        ks_stats.append((ks_stat, p_value))

    return pd.DataFrame(ks_stats, index=list(data1.columns), columns=['KS-Stats', 'P-Value'])


if __name__ == '__main__':
    data1 = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
    data2 = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])

    ks_stats = calculate_ks_statistics_by_feature(data1, data2)
    print(ks_stats)

