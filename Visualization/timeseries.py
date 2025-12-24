import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf
from scipy.signal import correlate


def vis_acf(data1: pd.DataFrame or np.array,
            label1: str=None,
            data2: pd.DataFrame or np.array=None,
            label2: str=None,
            lags: int=30,
            figsize: tuple=(8, 6),
            save_path: str=None,
            grid: bool=True,
            title: str=None,
            xlabel: str='Time lag',
            ylabel: str='Autocorrelation') -> None:
    acf_data1 = acf(data1, nlags=lags, missing='drop')
    acf_data2 = acf(data2, nlags=lags, missing='drop') if data2 is not None else None

    rmse = np.round(np.sqrt(np.mean((acf_data1 - acf_data2) ** 2)), 3)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)
    plt.plot(range(len(acf_data1)), acf_data1, 'o-', label=label1, color='royalblue')
    plt.fill_between(range(len(acf_data1)), acf_data1 - 0.1, acf_data1 + 0.1, color='royalblue', alpha=0.2)

    if data2 is not None:
        plt.plot(range(len(acf_data2)), acf_data2, 'o-', label=label2, color='chocolate')
        plt.fill_between(range(len(acf_data2)), acf_data2 - 0.1, acf_data2 + 0.1, color='chocolate', alpha=0.2)

    plt.text(lags-8, 0.9, f'RMSE = {rmse:.4f}', fontsize=12)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if (label1 is not None) or (label2 is not None):
        plt.legend(loc='upper right')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def vis_trajectory(data: pd.DataFrame or List[pd.DataFrame],
                   label: str or List[str] = None,
                   figsize: tuple=(8, 6),
                   x_col: str=None,
                   y_col: str=None,
                   save_path: str=None,
                   grid: bool=True,
                   title: str=None,
                   xlabel: str=None,
                   ylabel: str=None,
                   xticks: np.array or List[int] or List[float]=None,
                   yticks: np.array or List[int] or List[float]=None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    if isinstance(data, list):
        if label is not None:
            assert len(data) == len(label), 'Data and label must have the same length'

            for i, d in enumerate(data):
                sns.lineplot(data=d, x=x_col, y=y_col,
                             estimator='mean', errorbar='sd',
                             label=label[i], ax=ax)
            plt.legend(loc='upper left')
        else:
            for d in data:
                sns.lineplot(data=d, x=x_col, y=y_col,
                             estimator='mean', errorbar='sd', ax=ax)
    else:
        if label is not None:
            sns.lineplot(data=data, x=x_col, y=y_col,
                         estimator='mean', errorbar='sd', label=label, ax=ax)
            plt.legend(loc='upper left')
        else:
            sns.lineplot(data=data, x=x_col, y=y_col,
                         estimator='mean', errorbar='sd', ax=ax)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def timeseries_trajectory(data1: pd.DataFrame or np.array,
                          label1: str=None,
                          data2: pd.DataFrame or np.array=None,
                          label2: str=None,
                          seq_len: int=20,
                          figsize: tuple=(8, 6),
                          save_path: str=None,
                          grid: bool=True,
                          title: str=None,
                          num_yticks: int=10,
                          xlabel: str='hours_in',
                          ylabel: str='value') -> None:
    data1_mean = np.nanmean(data1, axis=0)
    data1_min = np.nanmin(data1, axis=0)
    data1_max = np.nanmax(data1, axis=0)
    data1_std = np.nanstd(data1, axis=0)
    data2_mean = np.nanmean(data2, axis=0) if data2 is not None else None
    data2_min = np.nanmin(data2, axis=0) if data2 is not None else None
    data2_max = np.nanmax(data2, axis=0) if data2 is not None else None
    data2_std = np.nanstd(data2, axis=0) if data2 is not None else None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)
    plt.plot(range(seq_len), data1_mean, linestyle='-', label=label1, color='royalblue')
    # plt.fill_between(range(seq_len), max(data1_min, data1_mean - data1_std), min(data1_mean + data1_std, data1_max), color='royalblue', alpha=0.2)
    plt.fill_between(range(seq_len), data1_mean - data1_std, data1_mean + data1_std, color='royalblue', alpha=0.2)
    plt.plot(range(seq_len), data2_mean, linestyle='-', label=label2, color='chocolate')
    # plt.fill_between(range(seq_len), max(data2_min, data2_mean - data2_std), min(data2_mean + data2_std, data2_max), color='chocolate', alpha=0.2)
    plt.fill_between(range(seq_len), data2_mean - data2_std, data2_mean + data2_std, color='chocolate', alpha=0.2)

    _min = min(np.nanmin(data1_mean - data1_std), np.nanmin(data2_mean - data2_std))
    _max = max(np.nanmax(data1_mean + data1_std), np.nanmax(data2_mean + data2_std))
    # _min = min(np.nanmin(data1), np.nanmin(data2))
    # _max = max(np.nanmax(data1), np.nanmax(data2))
    # plt.yticks(list(range((int(_min)//yticks_interval)*yticks_interval, (int(_max)//yticks_interval + 1)*yticks_interval, yticks_interval)))
    plt.yticks(np.linspace(_min, _max, num=num_yticks))

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.legend(loc='upper left')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def vis_cross_correlation(cross_corr,
                          lags: int=20,
                          figsize: tuple=(8, 6),
                          title: str='Cross-Correlation',
                          xlabel: str='Lag',
                          ylabel: str='Cross-Correlation',
                          save_path: str=None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=True)
    plt.plot(lags, cross_corr)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":

    real_data = np.random.randn(512, 20)
    synthetic_data = np.random.randn(512, 20)

    timeseries_trajectory(data1=real_data, label1='Real', data2=synthetic_data, label2='Synthetic')

    # data = pd.DataFrame(np.random.randn(100, 1), columns=['value'])
    # vis_acf(data1=real_data, label1='Real', data2=synthetic_data, label2='Synthetic', lags=30)

    # data = pd.DataFrame(np.random.randn(100, 2), columns=['value', 'time'])
    # data['time'] = data['time'].astype(int)
    # vis_trajectory(data, x_col='time', y_col='value')