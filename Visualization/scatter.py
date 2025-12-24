import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def scatterplot_dimension_wise_probability(data1: pd.DataFrame,
                                           data2: pd.DataFrame,
                                           label1: str,
                                           label2: str,
                                           figsize: tuple = (6, 6),
                                           save_path: str = None,
                                           grid: bool = True,
                                           title: str = None,
                                           xlabel: str = None,
                                           ylabel: str = None) -> None:
    def _correlation_coefficient(x, y):
        return np.corrcoef(x, y)[0, 1]

    def _root_mean_squared_error(x, y):
        return np.sqrt(np.mean((x - y) ** 2))

    x = data1.copy()
    y = data2.copy()

    x_prob = np.mean(x.values, axis=0)
    y_prob = np.mean(y.values, axis=0)
    feature = np.concatenate([[i] for i in range(x.shape[-1])], axis=0)
    df = pd.DataFrame({'x': x_prob, 'y': y_prob, 'feature': feature})

    cc = _correlation_coefficient(x_prob.tolist(), y_prob.tolist())
    rmse = _root_mean_squared_error(x_prob, y_prob)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)
    sns.scatterplot(x='x', y='y', hue='feature', data=df,
                    s=18, alpha=0.8, edgecolor='none', legend=None, palette='Paired_r')
    sns.lineplot(ax=ax, x=[0, 1.05], y=[0, 1.05], linewidth=0.5, color="red")

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
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.text(0.73, 0.1, f'CC={cc:.3f}', fontsize=9)
    ax.text(0.73, 0.05, f'RMSE={rmse:.3f}', fontsize=9)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    data1 = pd.DataFrame({
        'x': np.random.choice([0, 1], 100),
        'y': np.random.choice([0, 1], 100)
    })
    data2 = pd.DataFrame({
        'x': np.random.choice([0, 1], 100),
        'y': np.random.choice([0, 1], 100)
    })
    scatterplot_dimension_wise_probability(data1, data2, label1='data1', label2='data2', title='Scatterplot')
