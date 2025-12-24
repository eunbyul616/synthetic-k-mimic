import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def histogram(data1: pd.DataFrame,
              label1: str=None,
              data2: pd.DataFrame=None,
              label2: str=None,
              bins: int=30,
              alpha: float=0.7,
              figsize: tuple=(8, 6),
              density: bool=True,
              grid: bool=True,
              title: str=None,
              xlabel: str=None,
              ylabel: str=None,
              save_path=None):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    plt.hist(x=data1, label=label1, color='royalblue', bins=bins, alpha=alpha, density=density)
    if data2 is not None:
        plt.hist(x=data2, label=label2, color='chocolate', bins=bins, alpha=alpha, density=density)

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


if __name__ == "__main__":
    import numpy as np

    data1 = pd.DataFrame({
        'x': np.random.randn(100)
    })

    data2 = pd.DataFrame({
        'x': np.random.randn(100)
    })

    histogram(data1, 'data1', data2, 'data2', title='Histogram')