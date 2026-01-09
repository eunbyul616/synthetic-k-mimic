import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


def scatterplot_dimension_wise_probability(data1: pd.DataFrame or np.array,
                                           data2: pd.DataFrame or np.array,
                                           label1: str,
                                           label2: str,
                                           figsize: tuple=(6, 6),
                                           save_path: str=None,
                                           grid: bool=False,
                                           alpha: float=0.8,
                                           palette: str='Paired_r',
                                           s: int=25,
                                           title: str=None,
                                           xlabel: str=None,
                                           ylabel: str=None) -> None:
    """
    Plot scatterplot of dimension-wise probability of two datasets.

    Args:
        data1: input data 1
        data2: input data 2
        label1: label of data 1
        label2: label of data 2
        figsize: size of figure.  Default=(8, 6).
        save_path: save the plot to the given path. If None, the plot will be displayed
        grid: If True, display the grid
        alpha: transparency of the fill. Default is 0.5
        palette: color palette of the plot. Default is 'Paired_r'
        s: size of the marker. Default is 20
        title: title of the plot. If None, no title will be displayed
        xlabel: xlabel of the plot. If None, default xlabel will be displayed
        ylabel: ylabel of the plot. If None, default ylabel will be displayed

    Returns:
        None
    """

    def _calculate_correlation_coefficient(x, y):
        return np.corrcoef(x, y)[0, 1]

    def _calculate_rmse(x, y):
        return np.sqrt(np.mean((x - y)**2))

    x = data1.copy()
    y = data2.copy()

    if isinstance(data1, pd.DataFrame):
        x = x.values
    if isinstance(data2, pd.DataFrame):
        y = y.values

    x_prob = np.mean(x, axis=0)
    y_prob = np.mean(y, axis=0)
    feature = np.arange(x.shape[1])
    df = pd.DataFrame({'x': x_prob, 'y': y_prob, 'feature': feature})

    x_prob = np.atleast_1d(x_prob).astype(float).ravel()
    y_prob = np.atleast_1d(y_prob).astype(float).ravel()

    cc = _calculate_correlation_coefficient(x_prob, y_prob)
    rmse = _calculate_rmse(x_prob, y_prob)
    err = np.abs(y_prob - x_prob)
    mean_abs_diff = np.mean(err)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)
    # sns.scatterplot(x='x', y='y', hue='feature', data=df,
    #                 s=s, alpha=alpha, edgecolor='none', legend=None, palette=palette)
    # sns.scatterplot(x='x', y='y', hue='feature', data=df,
    #                 c=distance, cmap='viridis',
    #                 s=s, alpha=alpha, edgecolor='none', legend=None)
    sns.lineplot(ax=ax, x=[0, 1.01], y=[0, 1.01], linewidth=0.5, color="red")

    tol1 = 0.01
    tol2 = 0.05
    xx = np.linspace(0, 1, 200)
    # ax.fill_between(xx, lower, upper, color='red', alpha=0.12,
    #                 label=f'|y - x| ≤ {tol:g}')
    plt.fill_between(xx, np.clip(xx - tol2, 0, 1), np.clip(xx + tol2, 0, 1),
                     color='red', alpha=0.1, label=rf'$|p_{{\mathrm{{real}}}} - p_{{\mathrm{{synth}}}}| ≤ {tol2:.2f}$')
    plt.fill_between(xx, np.clip(xx - tol1, 0, 1), np.clip(xx + tol1, 0, 1),
                     color='red', alpha=0.15, label=rf'$|p_{{\mathrm{{real}}}} - p_{{\mathrm{{synth}}}}| ≤ {tol1:.2f}$')

    inner = err <= tol1
    middle = (err > tol1) & (err <= tol2)
    outer = err > tol2

    ax.scatter(x_prob[inner], y_prob[inner],
               c='silver', s=s, alpha=alpha,
               # label=f'|y - x| ≤ {tol1}',
               label=rf'$|p_{{\mathrm{{real}}}} - p_{{\mathrm{{synth}}}}| ≤ {tol1:.2f}$',
               zorder=3)
    ax.scatter(x_prob[middle], y_prob[middle],
               c='orange', s=s, alpha=alpha,
               # label=f'{tol1} < |y - x| ≤ {tol2}',
               label=rf'${tol1:.2f} < |p_{{\mathrm{{real}}}} - p_{{\mathrm{{synth}}}}| ≤ {tol2:.2f}$',
               zorder=3)

    ax.scatter(x_prob[outer], y_prob[outer],
               c='red', s=s, edgecolor='k', alpha=alpha,
               # label=f'|y - x| > {tol2}',
               label=rf'$|p_{{\mathrm{{real}}}} - p_{{\mathrm{{synth}}}}| > {tol2:.2f}$',
               zorder=4)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(f"Bernoulli success probability of {label1}")

    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(f"Bernoulli success probability of {label2}")

    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.set_xlim([0, 1.01])
    ax.set_ylim([0, 1.01])
    plt.gca().set_aspect('equal', 'box')

    ax.text(0.68, 0.15, f'CC={cc:.3f}', fontsize=8, transform=ax.transAxes)
    ax.text(0.68, 0.11, f'RMSE={rmse:.3f}', fontsize=8, transform=ax.transAxes)
    ax.text(0.68, 0.07, rf'Mean $|p_{{\mathrm{{real}}}} - p_{{\mathrm{{synth}}}}| = {mean_abs_diff:.3f}$', fontsize=8, transform=ax.transAxes)

    plt.legend(fontsize=8, loc='upper left')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    data1 = np.random.binomial(1, 0.5, (100, 10))
    data2 = np.random.binomial(1, 0.5, (100, 10))
    scatterplot_dimension_wise_probability(data1, data2, 'data1', 'data2')