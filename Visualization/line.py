import matplotlib.pyplot as plt
import seaborn as sns


def compare_original_reconstructed(x,
                                   x_hat,
                                   figsize: tuple=(8, 6),
                                   save_path: str=None,
                                   grid: bool=True,
                                   title: str=None,
                                   marker: str=None,
                                   yticks=None,
                                   xticks=None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    if marker is not None:
        plt.plot(x, label='original', marker=marker)
        plt.plot(x_hat, label='reconstructed', marker=marker)
    else:
        plt.plot(x, label='original')
        plt.plot(x_hat, label='reconstructed')

    plt.legend(loc='upper left')

    if yticks is not None:
        plt.yticks(yticks)
    if xticks is not None:
        plt.xticks(xticks)
    if title is not None:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    sns.set(style="whitegrid")
    x = [1, 2, 3, 4, 5]
    x_hat = [1.1, 2.2, 3.3, 4.4, 5.5]
    compare_original_reconstructed(x, x_hat, marker='o', title='Original vs Reconstructed')
