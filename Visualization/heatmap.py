import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency



def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1)) if n > 1 else 0
    rcorr = r - ((r - 1)**2) / (n - 1) if n > 1 else r
    kcorr = k - ((k - 1)**2) / (n - 1) if n > 1 else k
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))) if min(kcorr - 1, rcorr - 1) > 0 else 0.0


def heatmap_correlation(data1: pd.DataFrame,
                        label1: str = None,
                        data2: pd.DataFrame = None,
                        label2: str = None,
                        plot_diff: bool = True,
                        annot: bool = True,
                        figsize: tuple = (12, 12),
                        save_path: str = None,
                        title: str = None,
                        threshold: float = 0.05,
                        sort: bool = True,
                        top_k: int = 20,
                        categorical=False) -> None:
    assert not plot_diff or data2 is not None, 'data2 must be provided when plot_diff is True'

    def compute_cramers_matrix(df):
        cols = df.columns
        n = len(cols)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mat[i, j] = cramers_v(df[cols[i]], df[cols[j]])
        return pd.DataFrame(mat, index=cols, columns=cols)

    def sort_by_similarity(corr_matrix: pd.DataFrame):
        from scipy.cluster.hierarchy import linkage, leaves_list
        linkage_result = linkage(corr_matrix, method='ward')
        leaf_order = leaves_list(linkage_result)
        return corr_matrix.iloc[leaf_order, :].iloc[:, leaf_order]

    def pick_vars_from_top_k_pairs(corr_matrix: pd.DataFrame, k: int, highest: bool = True):
        cm = corr_matrix.abs().copy()
        np.fill_diagonal(cm.values, 0)
        stacked = cm.stack()
        stacked = stacked.sort_values(ascending=not highest)
        top_pairs = stacked.head(k)
        vars_ = set()
        for (v1, v2) in top_pairs.index:
            vars_.add(v1)
            vars_.add(v2)
        return list(vars_)

    # 1) 전체 correlation
    if categorical:
        data1_corr_full = compute_cramers_matrix(data1).fillna(0.0)
        np.fill_diagonal(data1_corr_full.values, 1.0)
    else:
        data1_corr_full = data1.corr().fillna(0.0)
        np.fill_diagonal(data1_corr_full.values, 1.0)

    if categorical:
        data2_corr_full = compute_cramers_matrix(data2).fillna(0.0)
        np.fill_diagonal(data2_corr_full.values, 1.0)

    else:
        data2_corr_full = data2.corr().fillna(0.0)
        np.fill_diagonal(data2_corr_full.values, 1.0)

    # 공통 변수로 맞추기
    common_vars = data1_corr_full.columns.intersection(data2_corr_full.columns)
    data1_corr_full = data1_corr_full.loc[common_vars, common_vars]
    data2_corr_full = data2_corr_full.loc[common_vars, common_vars]

    # 2) 전체 기준으로 diff / mad / coverage 계산
    _diff_full = data1_corr_full - data2_corr_full
    p_full = _diff_full.shape[0]
    idx_full = np.triu_indices(p_full, k=1)
    diff_vec = _diff_full.values[idx_full]

    mad = np.mean(np.abs(diff_vec))
    tau = 0.05
    coverage = np.mean(np.abs(diff_vec) <= tau)

    # Correlation Sign Accuracy
    sign_match = np.sign(data1_corr_full) == np.sign(data2_corr_full)
    diag_mask = np.eye(sign_match.shape[0], dtype=bool)
    valid = ~diag_mask
    cor_acc = np.nanmean(sign_match[valid].astype(float))
    cor_acc = cor_acc * 100

    # Mean Absolute Correlation Difference
    diff = np.abs(data1_corr_full - data2_corr_full)
    mu_abs = np.nanmean(diff[valid])

    top_vars = pick_vars_from_top_k_pairs(data1_corr_full, top_k, highest=True)
    bottom_vars = pick_vars_from_top_k_pairs(data1_corr_full, top_k, highest=False)

    data1_top = data1_corr_full.loc[top_vars, top_vars]
    data2_top = data2_corr_full.loc[top_vars, top_vars]

    data1_bottom = data1_corr_full.loc[bottom_vars, bottom_vars]
    data2_bottom = data2_corr_full.loc[bottom_vars, bottom_vars]

    if sort:
        data1_top = sort_by_similarity(data1_top)
        data2_top = data2_top.loc[data1_top.index, data1_top.columns]

        data1_bottom = sort_by_similarity(data1_bottom)
        data2_bottom = data2_bottom.loc[data1_bottom.index, data1_bottom.columns]


    n_vars = data1_corr_full.shape[0]
    if n_vars > top_k:
        ncols = 3 if plot_diff else 2
        fig, axes = plt.subplots(2, ncols, figsize=figsize)

        # 윗줄: top-k
        sns.heatmap(data1_top, annot=annot, vmin=-1, vmax=1, fmt=".2f",
                    cmap='coolwarm', center=0, ax=axes[0, 0], cbar=False, square=True)
        axes[0, 0].set_title(f'{label1} (top-{top_k})\n', fontsize=14)

        sns.heatmap(data2_top, annot=annot, vmin=-1, vmax=1, fmt=".2f",
                    cmap='coolwarm', center=0, ax=axes[0, 1], cbar=False, square=True)
        axes[0, 1].set_title(f'{label2} (top-{top_k})\n', fontsize=14)

        if plot_diff:
            diff_top = (data1_top - data2_top).round(3)
            sns.heatmap(diff_top, annot=annot, vmin=-1, vmax=1, fmt=".2f",
                        cmap='coolwarm', center=0, ax=axes[0, 2], cbar=False, square=True)
            axes[0, 2].set_title(f'Difference (top-{top_k})\n', fontsize=14)

        # 아랫줄: bottom-k
        sns.heatmap(data1_bottom, annot=annot, vmin=-1, vmax=1, fmt=".2f",
                    cmap='coolwarm', center=0, ax=axes[1, 0], cbar=False, square=True)
        axes[1, 0].set_title(f'{label1} (bottom-{top_k})', fontsize=14)

        sns.heatmap(data2_bottom, annot=annot, vmin=-1, vmax=1, fmt=".2f",
                    cmap='coolwarm', center=0, ax=axes[1, 1], cbar=False, square=True)
        axes[1, 1].set_title(f'{label2} (bottom-{top_k})', fontsize=14)

        if plot_diff:
            diff_bottom = (data1_bottom - data2_bottom).round(3)
            sns.heatmap(diff_bottom, annot=annot, vmin=-1, vmax=1, fmt=".2f",
                        cmap='coolwarm', center=0, ax=axes[1, 2], cbar=False, square=True)
            axes[1, 2].set_title(f'Difference (bottom-{top_k})', fontsize=14)

        fig.tight_layout(rect=[0, 0, 0.95, 1])
        fig.subplots_adjust(left=0.10)
        cbar = fig.colorbar(
            axes[0, 2].collections[0],
            ax=axes.ravel().tolist(),
            fraction=0.015,
            pad=0.01,
        )

        for ax in axes.ravel():
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

        for ax in [axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]:
            ax.set_yticklabels([])
            ax.set_ylabel('')

        if title is None:
            fig.suptitle(f'Pearson Pairwise Correlations\nMAD: {mu_abs:.3f}, CorSignAcc: {cor_acc:.2f}%',
                         fontsize=16)
        else:
            fig.suptitle(title)

        if save_path is not None:
            fig.savefig(save_path, dpi=300)
        else:
            plt.show()

    else:
        ncols = 3 if plot_diff else 2
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        sns.heatmap(data1_corr_full, annot=annot, vmin=-1, vmax=1, fmt=".2f",
                    cmap='coolwarm', center=0, ax=axes[0], cbar=False, square=True)
        axes[0].set_title(f'{label1}\n', fontsize=14)
        sns.heatmap(data2_corr_full, annot=annot, vmin=-1, vmax=1, fmt=".2f",
                    cmap='coolwarm', center=0, ax=axes[1], cbar=False, square=True)
        axes[1].set_title(f'{label2}\n', fontsize=14)
        if plot_diff:
            diff_full = (data1_corr_full - data2_corr_full).round(3)
            sns.heatmap(diff_full, annot=annot, vmin=-1, vmax=1, fmt=".2f",
                        cmap='coolwarm', center=0, ax=axes[2], cbar=False, square=True)
            axes[2].set_title(f'Difference\n', fontsize=14)

        fig.tight_layout(rect=[0, 0, 0.95, 1])
        fig.subplots_adjust(left=0.10)
        cbar = fig.colorbar(
            axes[2].collections[0],
            ax=axes.ravel().tolist(),
            fraction=0.015,
            pad=0.01,
        )
        for ax in axes.ravel():
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
        for ax in [axes[1], axes[2]]:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        if title is None:
            if categorical:
                fig.suptitle(f'Cramér’s V Pairwise Associations\nMAD: {mu_abs:.3f}, CorSignAcc: {cor_acc:.2f}%', fontsize=16)
            else:
                fig.suptitle(f'Pearson Pairwise Correlations\nMAD: {mu_abs:.3f}, CorSignAcc: {cor_acc:.2f}%', fontsize=16)
        else:
            fig.suptitle(title)
        if save_path is not None:
            fig.savefig(save_path, dpi=300)
        else:
            plt.show()

    plt.close()


def heatmap_correlation_accuracy(data1: pd.DataFrame,
                                 data2: pd.DataFrame,
                                 label1: str=None,
                                 label2: str=None,
                                 annot: bool = True,
                                 figsize: tuple=(12, 6),
                                 save_path: str=None) -> None:

    def _discretize_corr(coeff):
        if coeff < -0.5:
            return 0  # strong negative
        elif coeff < -0.3:
            return 1  # middle negative
        elif coeff < -0.1:
            return 2  # low negative
        elif coeff < 0.1:
            return 3  # no correlation
        elif coeff < 0.3:
            return 4  # low positive
        elif coeff < 0.5:
            return 5  # middle positive
        else:
            return 6  # strong positive

    data1_corr = data1.corr()
    data2_corr = data2.corr()

    data1_discretized = data1_corr.applymap(_discretize_corr)
    data2_discretized = data2_corr.applymap(_discretize_corr)

    diff_abs = np.mean(np.abs(data1_corr - data2_corr))
    matching_pairs = np.sum((data1_discretized == data2_discretized).values)
    total_pairs = data1_corr.size
    corr_acc = matching_pairs / total_pairs

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(data1_corr, annot=annot, vmin=-1.0, vmax=1.0, fmt=".2f", cmap='coolwarm', center=0, ax=axes[0])
    axes[0].set_title(label1)
    sns.heatmap(data2_corr, annot=annot, vmin=-1.0, vmax=1.0, fmt=".2f", cmap='coolwarm', center=0, ax=axes[1])
    axes[1].set_title(label2)

    plt.suptitle(f'Pearson Pairwise Correlation\n μabs: {diff_abs:.2f}, CorAcc: {corr_acc:.2f}')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def vis_group_distance(group_distance1: pd.DataFrame,
                       label1: str=None,
                       group_distance2: pd.DataFrame=None,
                       label2: str=None,
                       plot_diff: bool=True,
                       figsize: tuple=(18, 6),
                       title: str=None,
                       save_path: str=None) -> None:

    group_distance1 = group_distance1.applymap(lambda x: np.round(x, 3))

    if group_distance2 is None:
        vmin = group_distance1.min().min()
        vmax = group_distance1.max().max()

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.heatmap(group_distance1, annot=True, vmin=vmin, vmax=vmax, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
        ax.set_title(label1)

    else:
        group_distance2 = group_distance2.applymap(lambda x: np.round(x, 3))
        vmin = min(group_distance1.min().min(), group_distance2.min().min())
        vmax = max(group_distance1.max().max(), group_distance2.max().max())

        if plot_diff:
            _diff = group_distance1 - group_distance2
            _diff = _diff.applymap(lambda x: np.round(x, 3))

            fig, ax = plt.subplots(1, 3, figsize=figsize)
            sns.heatmap(group_distance1, annot=True, vmin=vmin, vmax=vmax, fmt=".3f", cmap='coolwarm', center=0, ax=ax[0])
            ax[0].set_title(label1)
            sns.heatmap(group_distance2, annot=True, vmin=vmin, vmax=vmax, fmt=".3f", cmap='coolwarm', center=0, ax=ax[1])
            ax[1].set_title(label2)
            sns.heatmap(_diff, annot=True, vmin=(-1)*vmax, vmax=vmax, fmt=".3f", cmap='coolwarm', center=0, ax=ax[2])
            ax[2].set_title('Difference')
        else:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            sns.heatmap(group_distance1, annot=True, vmin=vmin, vmax=vmax, fmt=".3f", cmap='coolwarm', center=0, ax=ax[0])
            ax[0].set_title(label1)
            sns.heatmap(group_distance2, annot=True, vmin=vmin, vmax=vmax, fmt=".3f", cmap='coolwarm', center=0, ax=ax[1])
            ax[1].set_title(label2)

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def heatmap_confusion_matrix(y_true,
                             y_pred,
                             title: str='Confusion Matrix',
                             xlabel: str='Predicted',
                             ylabel: str='True',
                             save_path: str=None):

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()




if __name__ == "__main__":
    real = np.random.rand(10, 5)
    synthetic = np.random.rand(10, 5)

    real = pd.DataFrame(real, columns=['a', 'b', 'c', 'd', 'e'])
    synthetic = pd.DataFrame(synthetic, columns=['a', 'b', 'c', 'd', 'e'])

    # heatmap_correlation(real, 'Real Data', synthetic, 'Synthetic Data',
    #                 plot_diff=True)
    heatmap_correlation_accuracy(real, synthetic,
                                 'Real Data', 'Synthetic Data')



