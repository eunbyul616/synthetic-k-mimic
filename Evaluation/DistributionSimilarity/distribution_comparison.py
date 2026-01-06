import os
import numpy as np
import pandas as pd
from typing import List

from Utils.utils import classify_variables_types
from Visualization.distribution import distribution_cdf, distribution_countplot
from Visualization.scatter import scatterplot_dimension_wise_probability


def visualize_distribution_comparison(real_data: pd.DataFrame,
                                      synthetic_data: pd.DataFrame,
                                      label: List[str] = ['Real', 'Synthetic'],
                                      fill: bool=True,
                                      alpha: float=0.5,
                                      save_path: str=None) -> None:
    """
    Visualize the distribution comparison between real and synthetic data

    Args:
        real_data: Real data
        synthetic_data: Synthetic data
        label: label for real and synthetic data
        fill: If True, fill in the space under the curve
        alpha: transparency of the fill. Default is 0.5
        save_path: save the plot to the given path. If None, the plot will be displayed

    Returns:
        None
    """
    col_types = dict()
    for col in real_data.columns:
        col_types[col] = classify_variables_types(real_data[col])

    numerical_features = [k for k, v in col_types.items() if v == 'Numerical']
    categorical_features = [k for k, v in col_types.items() if v == 'Categorical']
    binary_features = [k for k, v in col_types.items() if v == 'Binary']

    # Distribution comparison for numerical features
    for col in numerical_features:
        if save_path is not None:
            fpath = os.path.join(save_path, f'Numerical_CDF_{col}.png')
        else:
            fpath = None

        distribution_cdf(data=[real_data[col], synthetic_data[col]],
                         label=label,
                         fill=fill,
                         alpha=alpha,
                         title=col,
                         save_path=fpath)

    # Distribution comparison for categorical features
    for col in categorical_features:
        if save_path is not None:
            fpath = os.path.join(save_path, f'Categorical_COUNTPLOT_{col}.png')
        else:
            fpath = None

        distribution_countplot(data=[real_data[col], synthetic_data[col]],
                               label=label,
                               title=col,
                               save_path=fpath)

    # Distribution comparison for binary features
    for col in binary_features:
        if save_path is not None:
            fpath = os.path.join(save_path, f'Binary_COUNTPLOT_{col}.png')
        else:
            fpath = None

        distribution_countplot(data=[real_data[col], synthetic_data[col]],
                               label=label,
                               title=col,
                               save_path=fpath)

    if save_path is not None:
        fpath = os.path.join(save_path, f'Binary_DWP.png')
    else:
        fpath = None

    scatterplot_dimension_wise_probability(data1=real_data[binary_features],
                                           data2=synthetic_data[binary_features],
                                           label1=label[0],
                                           label2=label[1],
                                           title='Dimension-wise probability (Binary Features)',
                                           save_path=fpath)


if __name__ == '__main__':
    import config_manager
    config_manager.load_config()
    cfg = config_manager.config

    save_path = cfg.path.plot_path

    real_data = pd.DataFrame({'A': np.random.normal(0, 1, 1000),
                              'B': np.random.normal(0, 1, 1000),
                              'C': np.random.choice(['X', 'Y', 'Z'], 1000),
                              'D': np.random.choice([0, 1], 1000),
                              'E': np.random.choice([0, 1], 1000)})

    synthetic_data = pd.DataFrame({'A': np.random.normal(0, 1, 1000),
                                   'B': np.random.normal(0, 1, 1000),
                                   'C': np.random.choice(['X', 'Y', 'Z'], 1000),
                                   'D': np.random.choice([0, 1], 1000),
                                   'E': np.random.choice([0, 1], 1000)})

    visualize_distribution_comparison(real_data=real_data,
                                      synthetic_data=synthetic_data,
                                      label=['Real', 'Synthetic'],
                                      fill=True,
                                      alpha=0.5,
                                      save_path=save_path)