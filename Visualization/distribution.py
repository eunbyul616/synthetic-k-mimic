import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


def distribution_pdf(data: pd.Series or np.array or List[pd.Series] or List[np.array],
                     label: str or List[str] = None,
                     figsize: tuple=(8, 6),
                     save_path: str=None,
                     grid: bool=True,
                     fill: bool=True,
                     alpha: float=0.5,
                     color: str='royalblue',
                     title: str=None,
                     xlabel: str=None,
                     ylabel: str=None) -> None:
    """
    Plot the probability density function of the given data.

    Args:
        data: input data
        label: label of the data.  If None, no label will be displayed
        figsize: size of the figure. Default is (8, 6)
        save_path: save the plot to the given path. If None, the plot will be displayed
        grid: If True, display the grid
        fill: If True, fill in the space under the curve
        alpha: transparency of the fill. Default is 0.5
        color: single color specification. Default is 'royalblue'
        title: title of the plot. If None, no title will be displayed
        xlabel: xlabel of the plot. If None, no xlabel will be displayed
        ylabel: ylabel of the plot. If None, no ylabel will be displayed

    Returns:
        None
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    fig.canvas.draw()

    if isinstance(data, list):
        if label is not None:
            assert len(data) == len(label), 'Data and label must have the same length'

            for i, d in enumerate(data):
                sns.kdeplot(d, label=label[i], fill=fill, alpha=alpha, color=color)
            plt.legend(loc='upper left')

        else:
            for d in data:
                sns.kdeplot(d, fill=fill, alpha=alpha, color=color)
    else:
        if label is not None:
            sns.kdeplot(data, label=label, fill=fill, alpha=alpha, color=color)
            plt.legend(loc='upper left')
        else:
            sns.kdeplot(data, fill=fill, alpha=alpha, color=color)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def distribution_cdf(data: pd.Series or np.array or List[pd.Series] or List[np.array],
                     label: str or List[str] = None,
                     figsize: tuple=(8, 6),
                     save_path: str=None,
                     grid: bool=True,
                     fill: bool=False,
                     alpha: float=0.5,
                     color='royalblue',
                     title: str=None,
                     xlabel: str=None,
                     ylabel: str=None) -> None:
    """
    Plot the cumulative distribution function of the given data.

    Args:
        data: input data
        label: label of the data.  If None, no label will be displayed
        figsize: size of the figure. Default is (8, 6)
        save_path: save the plot to the given path. If None, the plot will be displayed
        grid: If True, display the grid
        fill: If True, fill in the space under the curve
        alpha: transparency of the fill. Default is 0.5
        color: single color specification. Default is 'royalblue'
        title: title of the plot. If None, no title will be displayed
        xlabel: xlabel of the plot. If None, no xlabel will be displayed
        ylabel: ylabel of the plot. If None, no ylabel will be displayed

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    fig.canvas.draw()

    if isinstance(data, list):
        if label is not None:
            assert len(data) == len(label), 'Data and label must have the same length'

            for i, d in enumerate(data):
                sns.kdeplot(d, label=label[i], fill=fill, alpha=alpha, cumulative=True, color=color)
            plt.legend(loc='upper left')

        else:
            for d in data:
                sns.kdeplot(d, fill=fill, alpha=alpha, cumulative=True, color=color)
    else:
        if label is not None:
            sns.kdeplot(data, label=label, fill=fill, alpha=alpha, cumulative=True, color=color)
            plt.legend(loc='upper left')
        else:
            sns.kdeplot(data, fill=fill, alpha=alpha, cumulative=True, color=color)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def distribution_countplot(data: pd.DataFrame or np.array or List[pd.DataFrame] or List[np.array],
                           col: str = 'data',
                           label: str or List[str] = None,
                           figsize: tuple=(8, 6),
                           save_path: str=None,
                           grid: bool=True,
                           title: str=None,
                           xlabel: str=None,
                           ylabel: str=None) -> None:
    """
    Plot the count plot of the given data.

    Args:
        data: input data
        col: column name of the data
        label: label of the data.  If None, no label will be displayed
        figsize: size of the figure. Default is (8, 6)
        save_path: save the plot to the given path. If None, the plot will be displayed
        grid: If True, display the grid
        title: title of the plot. If None, no title will be displayed
        xlabel: xlabel of the plot. If None, no xlabel will be displayed
        ylabel: ylabel of the plot. If None, no ylabel will be displayed

    Returns:
        None
    """

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[col])

    if isinstance(data, list):
        for i, d in enumerate(data):
            if isinstance(d, np.ndarray):
                data[i] = pd.DataFrame(d, columns=[col])
            elif isinstance(d, pd.Series):
                data[i] = pd.DataFrame(d, columns=[col])
            else:
                data[i] = d

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    fig.canvas.draw()

    if isinstance(data, list):
        if label is not None:
            assert len(data) == len(label), 'Data and label must have the same length'

        if len(data) > 1:
            df = []
            for i, d in enumerate(data):
                if label is not None:
                    d['label'] = label[i]
                else:
                    d['label'] = i
                df.append(d)
            df = pd.concat(df)
            sns.countplot(data=df, x=col, hue='label')
        else:
            # if only one data is given
            for i, d in enumerate(data):
                if label is not None:
                    sns.countplot(data=d, x=col, label=label[i])
                else:
                    sns.countplot(data=d, x=col, label=i)

    else:
        if label is not None:
            sns.countplot(data, label=label)
        else:
            sns.countplot(data)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.tight_layout()
    plt.legend(loc='upper left')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()



def vis_pdf(data: pd.DataFrame or np.array or list,
            label: str or list = None,
            figsize: tuple=(8, 6),
            save_path: str=None,
            grid: bool=True,
            title: str=None,
            xlabel: str=None,
            ylabel: str=None) -> None:

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    fig.canvas.draw()

    if isinstance(data, list):
        if label is not None:
            assert len(data) == len(label), 'Data and label must have the same length'

            for i, d in enumerate(data):
                sns.kdeplot(d, label=label[i], fill=True, alpha=0.5)
            plt.legend(loc='upper left')

        else:
            for d in data:
                sns.kdeplot(d, fill=True, alpha=0.5)
    else:
        if label is not None:
            sns.kdeplot(data, label=label, fill=True, alpha=0.5)
            plt.legend(loc='upper left')
        else:
            sns.kdeplot(data, fill=True, alpha=0.5)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def vis_cdf(data: pd.DataFrame or np.array or List[pd.DataFrame] or List[np.array],
            label: str or List[str] = None,
            figsize: tuple=(8, 6),
            save_path: str=None,
            grid: bool=True,
            title: str=None,
            xlabel: str=None,
            ylabel: str=None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    fig.canvas.draw()

    if isinstance(data, list):
        if label is not None:
            assert len(data) == len(label), 'Data and label must have the same length'

            for i, d in enumerate(data):
                sns.kdeplot(d, label=label[i], cumulative=True)
            plt.legend(loc='upper left')

        else:
            for d in data:
                sns.kdeplot(d, cumulative=True)
    else:
        if label is not None:
            sns.kdeplot(data, label=label, cumulative=True)
            plt.legend(loc='upper left')
        else:
            sns.kdeplot(data, cumulative=True)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def distribution_pdf_by_group(
        data1: pd.DataFrame,
        group_col: str,
        value_col: str,
        data2: pd.DataFrame = None,
        label1: str = None,
        label2: str = None,
        figsize: tuple=(8, 6),
        save_path: str=None,
        grid: bool=True,
        title: str=None) -> None:

    if data2 is not None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        fig.canvas.draw()

        sns.kdeplot(data=data1, x=value_col, hue=group_col, ax=ax[0])
        ax[0].grid(visible=grid)
        if label1 is not None:
            ax[0].set_title(label1)

        sns.kdeplot(data=data2, x=value_col, hue=group_col, ax=ax[1])
        ax[1].grid(visible=grid)
        if label2 is not None:
            ax[1].set_title(label2)

        if title is not None:
            plt.suptitle(title)

    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.grid(visible=grid)
        sns.kdeplot(data=data1, x=value_col, hue=group_col)

        if title is not None:
            plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def distribution_cdf_by_group(
        data1: pd.DataFrame,
        group_col: str,
        value_col: str,
        data2: pd.DataFrame = None,
        label1: str = None,
        label2: str = None,
        figsize: tuple=(8, 6),
        save_path: str=None,
        grid: bool=True,
        title: str=None) -> None:

    if data2 is not None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        fig.canvas.draw()

        sns.kdeplot(data=data1, x=value_col, cumulative=True, hue=group_col, ax=ax[0])
        ax[0].grid(visible=grid)
        if label1 is not None:
            ax[0].set_title(label1)

        sns.kdeplot(data=data2, x=value_col, cumulative=True, hue=group_col, ax=ax[1])
        ax[1].grid(visible=grid)
        if label2 is not None:
            ax[1].set_title(label2)

        if title is not None:
            plt.suptitle(title)

    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.grid(visible=grid)
        sns.kdeplot(data=data1, x=value_col,  cumulative=True, hue=group_col)

        if title is not None:
            plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    data = np.random.randn(1000)
    # test distribution_pdf
    distribution_pdf(data, title='Test Distribution PDF')

    # test distribution_cdf
    distribution_cdf(data, title='Test Distribution CDF')

    # test distribution_countplot
    data1 = pd.Series(np.random.choice(['A', 'B', 'C'], 1000))
    data2 = pd.Series(np.random.choice(['A', 'B', 'C'], 1000))
    distribution_countplot([data1, data2], label=['Data1', 'Data2'], title='Test Distribution Countplot')