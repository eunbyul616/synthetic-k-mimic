import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def vis_tsne(data1: pd.DataFrame or pd.Series or np.array,
             label1: str=None,
             data2: pd.DataFrame or pd.Series or np.array=None,
             label2: str=None,
             perplexity: int=10,
             figsize: tuple=(8, 8),
             save_path: str=None,
             grid: bool=True,
             title: str=None,
             alpha: float=0.3) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    tsne = TSNE(n_components=2, perplexity=perplexity)
    data1_2d = tsne.fit_transform(data1)
    plt.scatter(data1_2d[:, 0], data1_2d[:, 1],  c='royalblue', alpha=alpha, label=label1)

    if data2 is not None:
        data2_2d = tsne.fit_transform(data2)
        plt.scatter(data2_2d[:, 0], data2_2d[:, 1], c='crimson', alpha=alpha, label=label2)

    if title is not None:
        plt.title(title)

    if (label1 is not None) or (label2 is not None):
        plt.legend(loc='upper right')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def vis_tsne_by_group(data1: np.array,
                      label1: str=None,
                      data2: np.array=None,
                      label2: str=None,
                      groups: np.array=None,
                      perplexity: int=10,
                      figsize: tuple=(6, 6),
                      save_path: str=None,
                      grid: bool=True,
                      title: str=None,
                      alpha: float=0.3) -> None:
    assert groups is not None, 'groups must be provided'
    tsne = TSNE(n_components=2, perplexity=perplexity)
    data1_2d = tsne.fit_transform(data1)
    labels = np.unique(groups)

    if data2 is not None:
        data2_2d = tsne.fit_transform(data2)

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        for label in labels:
            idx = groups == label

            ax[0].scatter(data1_2d[idx, 0], data1_2d[idx, 1], alpha=alpha, label=label)
            ax[1].scatter(data2_2d[idx, 0], data2_2d[idx, 1], alpha=alpha, label=label)

        ax[0].set_title(label1)
        ax[1].set_title(label2)
        ax[0].grid(visible=grid)
        ax[1].grid(visible=grid)

        if title is not None:
            plt.suptitle(title)

    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.grid(visible=grid)

        labels = np.unique(groups)
        for label in labels:
            idx = groups == label
            plt.scatter(data1_2d[idx, 0], data1_2d[idx, 1], alpha=alpha, label=label)

        if title is not None:
            plt.title(title)

    plt.legend(loc='upper right')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # data1 = np.random.rand(100, 10)
    # data2 = np.random.rand(100, 10)
    #
    # vis_tsne(data1, 'data1', data2, 'data2', title='t-SNE')

    data1 = np.random.rand(100, 10)
    data2 = np.random.rand(100, 10)
    groups = np.random.choice([0, 1, 2], 100)
    vis_tsne_by_group(data1, 'data1',  data2, 'data2', groups=groups, title='t-SNE by group',
                      figsize=(12, 6))