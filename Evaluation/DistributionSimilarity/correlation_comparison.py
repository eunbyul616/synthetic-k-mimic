import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Visualization.heatmap import heatmap_correlation


def pearson_pairwise_correlation_comparison(real, reconstructed, plot_file_path,
                                            figsize: tuple=(18, 6), categorical: bool=False):
    heatmap_correlation(data1=real,
                        data2=reconstructed,
                        label1='Real', label2='Synthetic', annot=False,
                        figsize=figsize,
                        categorical=categorical,
                        save_path=plot_file_path)



def nan_aware_corrcoef(X):
    D = X.shape[1]
    corr_matrix = np.full((D, D), np.nan)

    for i in range(D):
        for j in range(D):
            xi = X[:, i]
            xj = X[:, j]

            valid = ~np.isnan(xi) & ~np.isnan(xj)
            if np.sum(valid) < 2:
                continue

            xi_valid = xi[valid]
            xj_valid = xj[valid]

            xi_mean = xi_valid.mean()
            xj_mean = xj_valid.mean()

            numerator = np.sum((xi_valid - xi_mean) * (xj_valid - xj_mean))
            denominator = np.sqrt(np.sum((xi_valid - xi_mean) ** 2) * np.sum((xj_valid - xj_mean) ** 2))
            if denominator == 0:
                corr = 0.0
            else:
                corr = numerator / denominator
            corr_matrix[i, j] = corr

    return corr_matrix

def compute_grouped_temporal_correlation(data, variable_names, interval=3):
    N, T, D = data.shape
    assert T % interval == 0, "T must be divisible by interval"
    K = T // interval  # Number of time groups

    features = []
    labels = []

    for d in range(D):
        for k in range(K):
            start = k * interval
            end = (k + 1) * interval
            segment = data[:, start:end, d]  # shape: (N, interval)
            segment_mean = np.nanmean(segment, axis=1)  # shape: (N,)
            features.append(segment_mean)
            labels.append(f"{variable_names[d]}_{start}-{end-1}")

    feature_matrix = np.stack(features, axis=1)  # shape: (N, D*K)
    corr_matrix = nan_aware_corrcoef(feature_matrix)

    return corr_matrix, labels


def compare_temporal_correlation(real_data, synth_data, variable_names, interval=3):
    corr_real, labels = compute_grouped_temporal_correlation(real_data, variable_names, interval)
    corr_synth, _ = compute_grouped_temporal_correlation(synth_data, variable_names, interval)

    # Correlation Sign Accuracy
    sign_match = np.sign(corr_real) == np.sign(corr_synth)
    cor_acc = np.nanmean(sign_match.astype(float))

    # Mean Absolute Correlation Difference
    mu_abs = np.nanmean(np.abs(corr_real - corr_synth))

    return corr_real, corr_synth, labels, cor_acc, mu_abs


def plot_correlation_matrices(cor_real, cor_synth, labels, cor_acc, mu_abs,
                              interval=10, figsize=(18, 6), mask_upper=True, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    simple_labels = [label.split('_')[0] for label in labels]

    ticks = list(range(len(simple_labels)))
    display_labels = [simple_labels[i] if i % interval == (interval//2) else '' for i in ticks]
    mask = np.triu(np.ones_like(cor_real, dtype=bool)) if mask_upper else None

    diff = cor_real - cor_synth
    for i, (ax, matrix, title) in enumerate(zip(
        axes,
        [cor_real, cor_synth, diff],
        ["Real", "Synthetic", 'Difference']
    )):
        sns.heatmap(matrix, xticklabels=display_labels, yticklabels=display_labels,
                    cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax, mask=mask, square=True, cbar=(i == len(axes)-1))
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=90, labelsize=7, length=0)
        ax.tick_params(axis='y', labelsize=7, length=0)

    for ax in [axes[1], axes[2]]:
        ax.set_yticklabels([])
        ax.set_ylabel('')

    plt.suptitle(f"Temporal Cross-sectional Correlation\nMAD: {mu_abs:.3f}, CorSignAcc: {cor_acc*100:.2f}%", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()