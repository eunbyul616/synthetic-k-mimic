import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset
from Utils.namespace import _load_yaml

from Evaluation.DistributionSimilarity.distribution_summary import *
from Evaluation.DistributionSimilarity.distribution_comparison import *
from Evaluation.DistributionSimilarity.correlation_comparison import *
from Evaluation.Fidelity.utils import *

from Visualization.timeseries import *

import config_manager


if __name__ == '__main__':
    config_manager.load_config()
    eval_cfg = config_manager.config

    cols = None
    dataset_name = eval_cfg.dataset.dataset_name
    fname = eval_cfg.preprocess.preprocess_fname_suffix
    train_ratio, test_ratio = eval_cfg.preprocess.train_valid_test_split_ratio
    dataset_fname = f'{dataset_name}_{fname}_{int(train_ratio * 10)}.h5'

    dataset = CustomDataset(cfg=eval_cfg,
                            dataset_name=dataset_name,
                            dataset_fname=dataset_fname,
                            mode='test',
                            condition_col=eval_cfg.data.condition_col,
                            static_cols=None)

    model_name = eval_cfg.evaluation.model_name
    checkpoint = eval_cfg.evaluation.checkpoint
    model_checkpoint_path = os.path.join('/'.join(Path(eval_cfg.path.ckpt_path).parts[:-1]), model_name, checkpoint)
    cfg = _load_yaml(os.path.join(model_checkpoint_path, 'config.yaml'))

    print(checkpoint)
    eval_file_path = os.path.join(eval_cfg.path.eval_file_path, model_name, checkpoint)
    print(eval_file_path)

    use_gumbel = True
    logit_threshold = 0.5
    seq_len = cfg.dataloader.seq_len

    static_transformer = dataset.static_transformer
    temporal_transformer = dataset.temporal_transformer

    sc_cols = dataset.sc_cols
    tc_cols = dataset.tc_cols
    sn_cols = dataset.sn_cols
    tn_cols = dataset.tn_cols
    sl_cols = dataset.sl_cols
    tl_cols = dataset.tl_cols

    diagnoses_prefix = cfg.preprocess.icd_code.diagnoses_prefix
    procedure_prefix = cfg.preprocess.icd_code.procedure_prefix
    proc_prefix = cfg.preprocess.proc_prefix

    static_data = pd.read_csv(os.path.join(eval_file_path, 'static_data.csv'))
    temporal_data = pd.read_csv(os.path.join(eval_file_path, 'temporal_data.csv'))
    static_data_hat = pd.read_csv(os.path.join(eval_file_path, 'static_reconstructed.csv'))
    temporal_data_hat = pd.read_csv(os.path.join(eval_file_path, 'temporal_reconstructed.csv'))

    icd_d_cols = [col for col in static_data.columns if (col.startswith(diagnoses_prefix)) and ('mask' not in col)]
    icd_p_cols = [col for col in static_data.columns if (col.startswith(procedure_prefix)) and ('mask' not in col)]

    static_cols = sn_cols + sc_cols
    static_df = static_data[static_cols].copy()
    static_mask_df = static_data[[f'{col}_mask' for col in static_cols]].copy()
    static_mask_df = static_mask_df.map(lambda x: 1 if x == 1 else np.nan)
    for col in static_cols:
        if np.issubdtype(static_df[col].dtype, np.number):
            static_df[col] = static_mask_df[f'{col}_mask'] * static_df[col]
        else:
            static_df[col] = static_df[col].where(static_mask_df[f'{col}_mask'].notna(), np.nan)

    static_df_hat = static_data_hat[static_cols].copy()
    static_mask_df_hat = static_data_hat[[f'{col}_mask' for col in static_cols]].copy()
    static_mask_df_hat = static_mask_df_hat.map(lambda x: 1 if x == 1 else np.nan)
    for col in static_cols:
        if np.issubdtype(static_df_hat[col].dtype, np.number):
            static_df_hat[col] = static_mask_df_hat[f'{col}_mask'] * static_df_hat[col]
        else:
            static_df_hat[col] = static_df_hat[col].where(static_mask_df_hat[f'{col}_mask'].notna(), np.nan)

    proc_cols = [col for col in temporal_data.columns if (col.startswith(proc_prefix)) and ('mask' not in col)]

    temporal_cols = tn_cols + tc_cols
    temporal_df = temporal_data[temporal_cols].copy()
    temporal_mask_cols = [f'{col}_mask' for col in temporal_cols]
    temporal_mask_df = temporal_data[temporal_mask_cols].copy()
    temporal_mask_df = temporal_mask_df.map(lambda x: 1 if x == 1 else np.nan)
    for col in temporal_cols:
        if np.issubdtype(temporal_df[col].dtype, np.number):
            temporal_df[col] = temporal_mask_df[f'{col}_mask'] * temporal_df[col]
        else:
            temporal_df[col] = temporal_df[col].where(temporal_mask_df[f'{col}_mask'].notna(), np.nan)

    temporal_df_hat = temporal_data_hat[temporal_cols].copy()
    temporal_mask_df_hat = temporal_data_hat[temporal_mask_cols].copy()
    temporal_mask_df_hat = temporal_mask_df_hat.map(lambda x: 1 if x == 1 else np.nan)
    for col in temporal_cols:
        if np.issubdtype(temporal_df_hat[col].dtype, np.number):
            temporal_df_hat[col] = temporal_mask_df_hat[f'{col}_mask'] * temporal_df_hat[col]
        else:
            temporal_df_hat[col] = temporal_df_hat[col].where(temporal_mask_df_hat[f'{col}_mask'].notna(), np.nan)

    save_eval_res_path = os.path.join(eval_file_path, 'evaluation')
    os.makedirs(save_eval_res_path, exist_ok=True)

    static_repeat = static_data.reset_index().rename(columns={'index': 'patient_id'})
    static_repeat = pd.DataFrame(np.repeat(static_repeat.values, seq_len, axis=0),
                                 columns=static_repeat.columns)
    total_data = pd.concat([static_repeat, temporal_data.reset_index(drop=True)], axis=1)
    total_data = total_data.drop(columns=['patient_id'])

    static_repeat_hat = static_data_hat.reset_index().rename(columns={'index': 'patient_id'})
    static_repeat_hat = pd.DataFrame(np.repeat(static_repeat_hat.values, seq_len, axis=0),
                                    columns=static_repeat_hat.columns)
    total_data_hat = pd.concat([static_repeat_hat, temporal_data_hat.reset_index(drop=True)], axis=1)
    total_data_hat = total_data_hat.drop(columns=['patient_id'])

    # Fidelity: cross-sectional consistency
    # 1) ks test
    static_consistency = compare_data_statistics(static_df[sn_cols], static_df_hat[sn_cols])
    temporal_consistency = compare_data_statistics(temporal_df[tn_cols], temporal_df_hat[tn_cols])
    consistency = pd.concat([static_consistency, temporal_consistency], axis=0)
    consistency[[('mean', 'Real'), ('std', 'Real'), ('missing_rate', 'Real'), ('mean', 'Synthetic'), ('std', 'Synthetic'), ('missing_rate', 'Synthetic'),  'KS-Stats', 'P-Value']].to_csv(os.path.join(save_eval_res_path, 'consistency.csv'), encoding='euc-kr')

    # 2) dimension-wise probability
    scatterplot_dimension_wise_probability(total_data[icd_d_cols+icd_p_cols+proc_cols].dropna(),
                                           total_data_hat[icd_d_cols+icd_p_cols+proc_cols].dropna(),
                                           'real',
                                           'synthetic',
                                           title='Binary Features Dimension-wise Probability',
                                           save_path=os.path.join(save_eval_res_path, 'dimension_wise_probability_bin.png'))

    mask_cols = [col for col in total_data.columns if 'mask' in col]
    scatterplot_dimension_wise_probability(total_data[mask_cols].dropna(),
                                           total_data_hat[mask_cols].dropna(),
                                           'real',
                                           'synthetic',
                                           title='Mask Dimension-wise Probability',
                                           save_path=os.path.join(save_eval_res_path, 'dimension_wise_probability_mask.png'))

    # 3) correlation
    pearson_pairwise_correlation_comparison(total_data[sn_cols + tn_cols],
                                            total_data_hat[sn_cols + tn_cols],
                                            categorical=False,
                                            figsize=(30, 22),
                                            plot_file_path=os.path.join(save_eval_res_path,
                                                                        'pearson_correlation.png'))
    pearson_pairwise_correlation_comparison(total_data[sc_cols + tc_cols],
                                            total_data_hat[sc_cols + tc_cols],
                                            categorical=True,
                                            figsize=(30, 11),
                                            plot_file_path=os.path.join(save_eval_res_path,
                                                                        'cramers_v_correlation.png'))


    # Fidelity: temporal consistency
    # 1) autocorrelation
    real = temporal_data[tn_cols].ffill().values.reshape(-1, seq_len, len(tn_cols))
    synth = temporal_data_hat[tn_cols].ffill().values.reshape(-1, seq_len, len(tn_cols))
    acf_df = compute_acf_table(real, synth, lags=[1, 12, 24, 29], feature_names=tn_cols)

    real_pivot = acf_df.pivot(index="feature", columns="lag", values="acf_real")
    synth_pivot = acf_df.pivot(index="feature", columns="lag", values="acf_synth")
    absdiff_pivot = acf_df.pivot(index="feature", columns="lag", values="abs_diff")

    feat_mean_err = absdiff_pivot.mean(axis=1)
    k = 20
    topk_features = feat_mean_err.sort_values(ascending=True).head(k)
    bottomk_features = feat_mean_err.sort_values(ascending=False).head(k)
    top10 = absdiff_pivot.loc[topk_features.index]
    bottom10 = absdiff_pivot.loc[bottomk_features.index]
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.heatmap(top10, annot=False, cmap="Reds", vmin=0, vmax=0.5, ax=ax[0], cbar=False)
    ax[0].set_title(f'Top {k} Features with Smallest ACF Differences')
    sns.heatmap(bottom10, annot=False, cmap="Reds", vmin=0, vmax=0.5, ax=ax[1])
    ax[1].set_title(f'Top {k} Features with Largest ACF Differences')

    abs_vals = absdiff_pivot.values
    overall_mad = abs_vals.mean()
    plt.suptitle("ACF abs difference between Real and Synthetic\n"
                 f"MAD = {overall_mad:.3f}")

    ax[0].set_ylabel("Feature Name")
    ax[0].set_xlabel("Lag (hours)")
    ax[1].set_ylabel("Feature Name")
    ax[1].set_xlabel("Lag (hours)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_eval_res_path, 'acf_mad.png'), dpi=300)

    # for col in tn_cols:
    #     _col = col.replace('/', ' ').replace('>', ' ')
    #     vis_acf(data1=temporal_data[[col]].dropna(), label1='Real', data2=temporal_data_hat[[col]].dropna(), label2='Synthetic',
    #             title=f'{col}',
    #             save_path=os.path.join(save_eval_res_path, f'acf_{_col}.png'))

    # 2) trajectory correlation
    # seq_len = 30
    # for col in tn_cols:
    #     _col = col.replace('/', ' ').replace('>', ' ')
    #     timeseries_trajectory(data1=temporal_df[col].ffill().values.reshape(-1, seq_len), label1='Real',
    #                           data2=temporal_df_hat[col].ffill().values.reshape(-1, seq_len), label2='Synthetic',
    #                           title=f'{col}',
    #                           seq_len=seq_len,
    #                           save_path=os.path.join(save_eval_res_path, f'trajectory_correlation_{_col}.png'))

    corr_matrix = total_data[sn_cols + tn_cols].corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs.sort_values(ascending=False)

    # threshold = 0.6
    # sig = corr_pairs[corr_pairs >= threshold]
    # top_vars = list(set([x for tup in sig.index for x in tup]))

    # top_10_pairs = corr_pairs.head(20)
    # top_vars = list(set(
    #     [idx for pair in top_10_pairs.index for idx in pair]
    # ))

    top_vars = tn_cols
    interval = 6
    corr_real, corr_synth, labels, cor_acc, mu_abs = compare_temporal_correlation(
        temporal_data[top_vars].values.reshape(-1, seq_len, len(top_vars)),
        temporal_data_hat[top_vars].values.reshape(-1, seq_len, len(top_vars)), top_vars, interval=interval
    )

    vars_ = [lab.split('_')[0] for lab in labels]
    times = [lab.split('_')[-1] for lab in labels]
    unique_times = sorted(set(times), key=lambda x: int(x.split('-')[0]))

    per_window_mu_abs = []
    per_win_std = []
    per_window_cor_acc = []
    per_window_time = []

    for t in unique_times:
        idx = [i for i, tt in enumerate(times) if tt == t]
        if len(idx) <= 1:
            continue

        real_sub = corr_real[np.ix_(idx, idx)]
        synth_sub = corr_synth[np.ix_(idx, idx)]

        n_sub = real_sub.shape[0]
        diag_mask = np.eye(n_sub, dtype=bool)
        valid = ~diag_mask

        diff_sub = np.abs(real_sub - synth_sub)
        mu_abs_sub = np.nanmean(diff_sub[valid])

        sign_match = np.sign(real_sub) == np.sign(synth_sub)
        cor_acc_sub = np.nanmean(sign_match[valid].astype(float))

        per_window_mu_abs.append(mu_abs_sub)
        per_win_std.append(np.nanstd(diff_sub[valid]))
        per_window_cor_acc.append(cor_acc_sub)
        per_window_time.append(t)

    mu_abs_mean = float(np.nanmean(per_window_mu_abs))
    mu_abs_std = float(np.nanstd(per_window_mu_abs))
    cor_acc_mean = float(np.nanmean(per_window_cor_acc))
    cor_acc_std = float(np.nanstd(per_window_cor_acc))

    plt.figure(figsize=(8, 4))
    plt.errorbar(
        per_window_time,
        per_window_mu_abs,
        yerr=per_win_std,
        fmt='-o',
        capsize=4,
    )
    plt.xlabel("Time Window (hours)")
    plt.ylabel("Mean Absolute Correlation Difference")
    plt.title("Per-window correlation error\n"
              f"MAD: {mu_abs_mean:.3f} ± {mu_abs_std:.3f}, CorSignAcc: {cor_acc_mean * 100:.2f}% ± {cor_acc_std * 100:.2f}%")
    plt.yticks(np.arange(0, 0.3, 0.05))
    plt.tight_layout()
    plt.savefig(os.path.join(save_eval_res_path, 'per_window_correlation_error.png'), dpi=300)

    # main vitals
    top_vars = ['SBP', 'DBP', 'MBP', 'HR', 'RR', 'SpO2', 'BT']
    corr_real, corr_synth, labels, cor_acc, mu_abs = compare_temporal_correlation(
        temporal_data[top_vars].values.reshape(-1, seq_len, len(top_vars)),
        temporal_data_hat[top_vars].values.reshape(-1, seq_len, len(top_vars)), top_vars, interval=interval
    )
    plot_correlation_matrices(corr_real, corr_synth, labels, cor_acc, mu_abs, figsize=(18, 6), interval=seq_len//interval,
                              save_path=os.path.join(save_eval_res_path, 'trajectory_correlation_comparison_main_vitals.png'))

    top_vars = ['Hb', 'Hct', 'RBC', 'MCV', 'MCH', 'PLT']
    corr_real, corr_synth, labels, cor_acc, mu_abs = compare_temporal_correlation(
        temporal_data[top_vars].values.reshape(-1, seq_len, len(top_vars)),
        temporal_data_hat[top_vars].values.reshape(-1, seq_len, len(top_vars)), top_vars, interval=interval
    )
    plot_correlation_matrices(corr_real, corr_synth, labels, cor_acc, mu_abs, figsize=(18, 6),
                              interval=seq_len // interval,
                              save_path=os.path.join(save_eval_res_path,
                                                     'trajectory_correlation_comparison_rbc.png'))

    top_10_pairs = corr_pairs.head(10)
    top_vars = list(set(
        [idx for pair in top_10_pairs.index for idx in pair]
    ))
    corr_real, corr_synth, labels, cor_acc, mu_abs = compare_temporal_correlation(
        temporal_data[top_vars].values.reshape(-1, seq_len, len(top_vars)),
        temporal_data_hat[top_vars].values.reshape(-1, seq_len, len(top_vars)), top_vars, interval=interval
    )
    plot_correlation_matrices(corr_real, corr_synth, labels, cor_acc, mu_abs, figsize=(18, 6),
                              interval=seq_len // interval,
                              save_path=os.path.join(save_eval_res_path,
                                                     'trajectory_correlation_comparison.png'))

    breakpoint()



