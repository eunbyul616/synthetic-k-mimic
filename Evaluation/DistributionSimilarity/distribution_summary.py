import os
import numpy as np
import pandas as pd
from typing import List

from Utils.metrics import calculate_ks_statistics_by_feature


def compare_data_statistics(real_data: pd.DataFrame,
                            synthetic_data: pd.DataFrame,
                            decimal: int=3,
                            label: List[str] = ['Real', 'Synthetic'],
                            summary_func: List[str] = ['mean', 'std'],
                            save_path: str=None) -> pd.DataFrame or None:
    """
    Compare data statistics between real and synthetic data

    Args:
        real_data: Real data
        synthetic_data: Synthetic data
        decimal: decimal point to round
        label: label for real and synthetic data
        summary_func: functions to summarize data. Default is ['mean', 'std']
        save_path: path to save the result

    Returns:
        pd.DataFrame or None
    """
    real_stats = real_data.describe().T
    synthetic_stats = synthetic_data.describe().T

    for func in summary_func:
        if func in real_stats.columns:
            continue

        real_stats[func] = real_data.apply(func, axis=0)
        synthetic_stats[func] = synthetic_data.apply(func, axis=0)

    real_stats = real_stats[summary_func]
    synthetic_stats = synthetic_stats[summary_func]

    real_stats = real_stats.applymap(lambda x: round(x, decimal))
    synthetic_stats = synthetic_stats.applymap(lambda x: round(x, decimal))

    real_missing_rate = pd.DataFrame(real_data.isnull().mean(), columns=['missing_rate'])
    synthetic_missing_rate = pd.DataFrame(synthetic_data.isnull().mean(), columns=['missing_rate'])
    real_stats = pd.concat([real_stats, real_missing_rate], axis=1)
    synthetic_stats = pd.concat([synthetic_stats, synthetic_missing_rate], axis=1)

    real_stats = real_stats.reset_index().rename(columns={'index': 'feature'})
    synthetic_stats = synthetic_stats.reset_index().rename(columns={'index': 'feature'})

    real_stats['data'] = label[0]
    synthetic_stats['data'] = label[1]

    stats = pd.concat([real_stats, synthetic_stats])
    ks_stats = calculate_ks_statistics_by_feature(real_data, synthetic_data, decimal=decimal)
    stats = pd.pivot_table(stats, index='feature', columns='data', values=summary_func+['missing_rate'])
    stats = pd.concat([stats, ks_stats], axis=1)

    if save_path is not None:
        fpath = os.path.join(save_path, 'data_statistics.csv')
        stats.to_csv(fpath)
    else:
        return stats


if __name__ == '__main__':
    real_data = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    real_data.loc[real_data.sample(frac=0.1).index, 'A'] = np.nan
    synthetic_data = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    synthetic_data.loc[synthetic_data.sample(frac=0.1).index, 'A'] = np.nan

    compare_data_statistics(real_data, synthetic_data)