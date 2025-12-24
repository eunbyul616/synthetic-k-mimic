import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


def boxplot(data: pd.DataFrame,
            x: str,
            y: str,
            hue: str=None,
            figsize: tuple=(8, 6),
            save_path: str=None,
            grid: bool=True,
            title: str=None,
            xlabel: str=None,
            ylabel: str=None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    if hue is not None:
        sns.boxplot(data=data, x=x, y=y, hue=hue)
    else:
        sns.boxplot(data=data, x=x, y=y)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.legend(loc='upper left')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    data = pd.DataFrame({
        'x': np.random.choice(['A', 'B', 'C'], 100),
        'y': np.random.randn(100),
        'hue': np.random.choice(['X', 'Y'], 100)
    })

    boxplot(data, 'x', 'y', 'hue', title='Boxplot', xlabel='X', ylabel='Y')