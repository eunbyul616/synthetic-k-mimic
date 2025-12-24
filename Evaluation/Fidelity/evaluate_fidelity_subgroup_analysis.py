import os
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from matplotlib.lines import Line2D

import config_manager
from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset
from Utils.namespace import _load_yaml
from Evaluation.DistributionSimilarity.distribution_summary import *
from Evaluation.DistributionSimilarity.distribution_comparison import *
from Evaluation.DistributionSimilarity.correlation_comparison import *
from Visualization.timeseries import *

from Evaluation.Fidelity.utils import build_cond_group

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

    try:
        z = np.load(os.path.join(eval_file_path, 'enc_z.npy'))
        z_hat = np.load(os.path.join(eval_file_path, 'enc_z_hat.npy'))
    except:
        z = None
        z_hat = None

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
    total_data['hours'] = total_data.groupby('patient_id').cumcount()
    total_data = total_data.drop(columns=['patient_id'])

    static_repeat_hat = static_data_hat.reset_index().rename(columns={'index': 'patient_id'})
    static_repeat_hat = pd.DataFrame(np.repeat(static_repeat_hat.values, seq_len, axis=0),
                                    columns=static_repeat_hat.columns)
    total_data_hat = pd.concat([static_repeat_hat, temporal_data_hat.reset_index(drop=True)], axis=1)
    total_data_hat['hours'] = total_data_hat.groupby('patient_id').cumcount()
    total_data_hat = total_data_hat.drop(columns=['patient_id'])

    static_data["cond_group"] = build_cond_group(static_data)
    static_data_hat["cond_group"] = build_cond_group(static_data_hat)
    total_data["cond_group"] = build_cond_group(total_data)
    total_data_hat["cond_group"] = build_cond_group(total_data_hat)

    group_order = ["ICU mortality", "Hospital mortality", "Alive"]
    palette = {
        "ICU mortality": "#D55E00",  # vermillion
        "Hospital mortality": "#0072B2",  # blue
        "Alive": "#009E73",  # bluish green
    }

    for col in sn_cols:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        # Real
        tmp = static_data[["cond_group", col]].dropna(subset=[col])

        sns.boxplot(
            data=tmp, x="cond_group", y=col,
            order=group_order, palette=palette, ax=ax[0]
        )
        ax[0].set_title("Real")
        ax[0].set_xlabel("")
        ax[0].set_ylabel(col)
        ax[0].tick_params(axis="x")

        # Synthetic
        tmp = static_data_hat[["cond_group", col]].dropna(subset=[col])
        sns.boxplot(
            data=tmp, x="cond_group", y=col,
            order=group_order, palette=palette, ax=ax[1]
        )
        ax[1].set_title("Synthetic")
        ax[1].set_xlabel("")
        ax[1].set_ylabel(col)
        ax[1].tick_params(axis="x")

        fig.suptitle(f"{col}")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(save_eval_res_path, f"boxplot_{col}.png"), dpi=300)
        plt.close(fig)

    cols = [
        'RR',
        'HR',
        'SBP',
        'DBP',
        'BT',
        'SpO2',
        'Glasgow coma scale(eye)',
        'Glasgow coma scale(verbal)',
        'Glasgow coma scale(motor)',
        'ALT',
        'AST',
        'Albumin',
        'BUN',
        'Bilirubin, total',
        'Chloride, serum',
        'Creatinine',
        'Glucose',
        'Hb',
        'PT (INR)',
        'PLT',
        'Potassium, serum',
        'Sodium, serum',
        'WBC',
        'aPTT'
    ]
    for col in cols:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        # Real
        tmp = total_data[["cond_group", col]].dropna(subset=[col])
        sns.boxplot(
            data=tmp, x="cond_group", y=col,
            order=group_order, palette=palette, ax=ax[0]
        )

        ax[0].set_title("Real")
        ax[0].set_xlabel("")
        ax[0].set_ylabel(col)
        ax[0].tick_params(axis="x")

        # Synthetic
        tmp = total_data_hat[["cond_group", col]].dropna(subset=[col])
        sns.boxplot(
            data=tmp, x="cond_group", y=col,
            order=group_order, palette=palette, ax=ax[1]
        )
        ax[1].set_title("Synthetic")
        ax[1].set_xlabel("")
        ax[1].set_ylabel(col)
        ax[1].tick_params(axis="x")

        fig.suptitle(f"{col}")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(save_eval_res_path, f"boxplot_{col}.png"), dpi=300)
        plt.close(fig)

    if z is not None and z_hat is not None:
        group_order = ["ICU mortality", "Hospital mortality", "Alive"]
        colors = sns.color_palette("colorblind", 3)
        palette = {
            "ICU mortality": "#D55E00",  # vermillion
            "Hospital mortality": "#0072B2",  # blue
            "Alive": "#009E73",  # bluish green
        }

        static_data["cond_group"] = build_cond_group(static_data)
        static_data_hat["cond_group"] = build_cond_group(static_data_hat)
        z_cols = [f"z{i}" for i in range(z.shape[1])]

        df_z = pd.DataFrame(z, columns=z_cols)
        df_z["cond_group"] = static_data["cond_group"].values
        df_z["source"] = "Real"

        df_zh = pd.DataFrame(z_hat, columns=z_cols)
        df_zh["cond_group"] = static_data_hat["cond_group"].values
        df_zh["source"] = "Synthetic"

        df_all = pd.concat([df_z, df_zh], ignore_index=True)

        X = df_all[z_cols].values
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42
        )
        umap_xy = reducer.fit_transform(X)

        df_all["UMAP1"] = umap_xy[:, 0]
        df_all["UMAP2"] = umap_xy[:, 1]

        g = sns.FacetGrid(
            df_all,
            col="source",
            hue="cond_group",
            hue_order=["ICU mortality", "Hospital mortality", "Alive"],
            palette=palette,
            height=5,
            aspect=1,
            sharex=True,
            sharey=True
        )
        g.map_dataframe(sns.scatterplot, x="UMAP1", y="UMAP2", s=18, alpha=0.6, linewidth=0)

        if g._legend is not None:
            g._legend.remove()

        handles = [
            Line2D([0], [0],
                   marker='o', linestyle='',
                   markerfacecolor=palette[k],
                   markeredgecolor='none',
                   markersize=6,
                   label=k)
            for k in ["ICU mortality", "Hospital mortality", "Alive"]
        ]
        g.fig.legend(handles=handles, title="Condition",
                     loc="center left", bbox_to_anchor=(0.80, 0.5), frameon=False)

        g.set_titles("{col_name}")
        g.set_axis_labels("UMAP1", "UMAP2")
        plt.tight_layout(rect=[0, 0, 0.80, 1])
        g.fig.savefig(os.path.join(save_eval_res_path, f"umap.png"), dpi=300)




