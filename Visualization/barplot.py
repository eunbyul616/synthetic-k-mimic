import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def countplot_categorical_feature(data1: pd.DataFrame,
                                  col: str,
                                  data2: pd.DataFrame = None,
                                  label1: str = None,
                                  label2: str = None,
                                  stat: str = 'count',
                                  figsize: tuple = (8, 6),
                                  grid: bool = True,
                                  title: str = None,
                                  save_path: str = None):
    assert stat in ['count', 'percent', 'proportion', 'probability'], 'stat should be one of count, percent, proportion, probability'

    x = data1.copy()
    y = data2.copy() if data2 is not None else None

    if y is not None:
        assert label1 is not None and label2 is not None, 'Please provide labels for both data1 and data2'

        x['label'] = label1
        y['label'] = label2
        data = pd.concat([x, y], axis=0, ignore_index=True)
    else:
        data = x

        if label1 is not None:
            data['label'] = label1

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    if label1 is not None:
        sns.countplot(data=data, x=col, hue='label', ax=ax, stat=stat)
    else:
        sns.countplot(data=data, x=col, ax=ax, stat=stat)

    if title is not None:
        plt.title(title)

    plt.xticks(rotation=90)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    x = pd.DataFrame({'feature': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']})
    y = pd.DataFrame({'feature': ['a', 'b', 'c', 'a', 'b', 'b', 'b', 'b', 'c', 'a', 'b', 'c']})

    countplot_categorical_feature(x, 'feature', y, 'x', 'y', title='Countplot of feature')