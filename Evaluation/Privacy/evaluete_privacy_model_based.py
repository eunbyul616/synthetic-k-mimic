import os
from pathlib import Path
import numpy as np
import pandas as pd

import config_manager

from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset

from Evaluation.Privacy.membership_inference import membership_inference_attack
from Evaluation.Privacy.re_identification import re_identification_attack
from Evaluation.Privacy.attribute_inference import attribute_inference_attack
from Evaluation.preprocess import *


def run_membership_attack(data, data_hat, static_cols, temporal_cols, seq_len, n_timestep):
    static_values = data[static_cols].iloc[np.arange(len(data), step=n_timestep)]
    static_hat_values = data_hat[static_cols].iloc[np.arange(len(data_hat), step=n_timestep)]
    temporal_values = data[temporal_cols]
    temporal_data_hat = data_hat[temporal_cols]

    # 1) membership inference attack
    train_indices, holdout_indices = train_holdout_split(data=static_values, split_ratio=0.5, random_seed=eval_cfg.seed)
    train_static_data = static_values.iloc[train_indices]
    holdout_static_data = static_values.iloc[holdout_indices]
    half_synthetic_static_data = static_hat_values.iloc[train_indices]
    rest_synthetic_static_data = static_hat_values.iloc[holdout_indices]

    train_static_data['type'] = 'train'
    holdout_static_data['type'] = 'holdout'
    static_real_data = pd.concat([train_static_data, holdout_static_data], axis=0).sort_index()
    static_real_data = static_real_data.reset_index().rename(columns={'index': 'patient_id'})
    static_repeat = pd.DataFrame(np.repeat(static_real_data[['patient_id', 'type']].values, seq_len, axis=0),
                                 columns=['patient_id', 'type'])

    temporal_train_indies = static_repeat[static_repeat['type'] == 'train'].index
    temporal_holdout_indies = static_repeat[static_repeat['type'] == 'holdout'].index
    temporal_data = data[temporal_cols]
    temporal_synthetic_data = data_hat[temporal_cols]
    temporal_real_data = temporal_data  # pd.concat([time_data, temporal_data], axis=1)
    temporal_synthetic_data = temporal_synthetic_data  # pd.concat([time_data_hat, temporal_synthetic_data], axis=1)
    train_temporal_data = temporal_real_data.iloc[temporal_train_indies]
    holdout_temporal_data = temporal_real_data.iloc[temporal_holdout_indies]
    half_synthetic_temporal_data = temporal_synthetic_data.iloc[temporal_train_indies]
    rest_synthetic_temporal_data = temporal_synthetic_data.iloc[temporal_holdout_indies]

    train_temporal_data = train_temporal_data.fillna(-1)
    holdout_temporal_data = holdout_temporal_data.fillna(-1)
    half_synthetic_temporal_data = half_synthetic_temporal_data.fillna(-1)
    rest_synthetic_temporal_data = rest_synthetic_temporal_data.fillna(-1)

    train_temporal_data = train_temporal_data.values.reshape(-1, seq_len, temporal_real_data.shape[-1])
    holdout_temporal_data = holdout_temporal_data.values.reshape(-1, seq_len, temporal_real_data.shape[-1])
    half_synthetic_data = half_synthetic_temporal_data.values.reshape(-1, seq_len, temporal_real_data.shape[-1])
    rest_synthetic_data = rest_synthetic_temporal_data.values.reshape(-1, seq_len, temporal_real_data.shape[-1])

    num_patients, seq_len, num_features = train_temporal_data.shape
    train_temporal_data = train_temporal_data.reshape(num_patients, seq_len * num_features)
    num_patients, seq_len, num_features = holdout_temporal_data.shape
    holdout_temporal_data = holdout_temporal_data.reshape(num_patients, seq_len * num_features)
    num_patients, seq_len, num_features = half_synthetic_data.shape
    half_synthetic_data = half_synthetic_data.reshape(num_patients, seq_len * num_features)
    num_patients, seq_len, num_features = rest_synthetic_data.shape
    rest_synthetic_data = rest_synthetic_data.reshape(num_patients, seq_len * num_features)

    exclude_cols = [col for col in static_real_data.columns if 'discharge_location' in col]
    train_static_data = train_static_data.drop(columns=['type'] + exclude_cols)
    holdout_static_data = holdout_static_data.drop(columns=['type'] + exclude_cols)
    half_synthetic_static_data = half_synthetic_static_data.drop(columns=exclude_cols)
    rest_synthetic_static_data = rest_synthetic_static_data.drop(columns=exclude_cols)

    missing_cols = list(half_synthetic_static_data.isnull().sum()[half_synthetic_static_data.isnull().sum() > 0].index)
    for col in missing_cols:
        half_synthetic_static_data[col] = half_synthetic_static_data[col].fillna(
            half_synthetic_static_data[col].mode()[0])
        rest_synthetic_static_data[col] = rest_synthetic_static_data[col].fillna(
            rest_synthetic_static_data[col].mode()[0])

    train_data = np.concatenate([train_static_data, train_temporal_data], axis=1)
    holdout_data = np.concatenate([holdout_static_data, holdout_temporal_data], axis=1)
    half_synthetic_data = np.concatenate([half_synthetic_static_data, half_synthetic_data], axis=1)
    rest_synthetic_data = np.concatenate([rest_synthetic_static_data, rest_synthetic_data], axis=1)
    acc = membership_inference_attack(train_data, holdout_data, half_synthetic_data)

    return acc


def run_reidentification_attack(data, data_hat, static_cols, temporal_cols, seq_len, n_timestep):
    static_values = data[static_cols].iloc[np.arange(len(data), step=n_timestep)]
    static_hat_values = data_hat[static_cols].iloc[np.arange(len(data_hat), step=n_timestep)]
    temporal_values = data[temporal_cols]
    temporal_data_hat = data_hat[temporal_cols]

    # 1) membership inference attack
    train_indices, holdout_indices = train_holdout_split(data=static_values, split_ratio=0.5, random_seed=eval_cfg.seed)
    train_static_data = static_values.iloc[train_indices]
    holdout_static_data = static_values.iloc[holdout_indices]
    half_synthetic_static_data = static_hat_values.iloc[train_indices]
    rest_synthetic_static_data = static_hat_values.iloc[holdout_indices]

    train_static_data['type'] = 'train'
    holdout_static_data['type'] = 'holdout'
    static_real_data = pd.concat([train_static_data, holdout_static_data], axis=0).sort_index()
    static_real_data = static_real_data.reset_index().rename(columns={'index': 'patient_id'})
    static_repeat = pd.DataFrame(np.repeat(static_real_data[['patient_id', 'type']].values, seq_len, axis=0),
                                 columns=['patient_id', 'type'])

    temporal_train_indies = static_repeat[static_repeat['type'] == 'train'].index
    temporal_holdout_indies = static_repeat[static_repeat['type'] == 'holdout'].index
    temporal_data = data[temporal_cols]
    temporal_synthetic_data = data_hat[temporal_cols]
    temporal_real_data = temporal_data  # pd.concat([time_data, temporal_data], axis=1)
    temporal_synthetic_data = temporal_synthetic_data  # pd.concat([time_data_hat, temporal_synthetic_data], axis=1)
    train_temporal_data = temporal_real_data.iloc[temporal_train_indies]
    holdout_temporal_data = temporal_real_data.iloc[temporal_holdout_indies]
    half_synthetic_temporal_data = temporal_synthetic_data.iloc[temporal_train_indies]
    rest_synthetic_temporal_data = temporal_synthetic_data.iloc[temporal_holdout_indies]

    train_temporal_data = train_temporal_data.fillna(-1)
    holdout_temporal_data = holdout_temporal_data.fillna(-1)
    half_synthetic_temporal_data = half_synthetic_temporal_data.fillna(-1)
    rest_synthetic_temporal_data = rest_synthetic_temporal_data.fillna(-1)

    train_temporal_data = train_temporal_data.values.reshape(-1, seq_len, temporal_real_data.shape[-1])
    holdout_temporal_data = holdout_temporal_data.values.reshape(-1, seq_len, temporal_real_data.shape[-1])
    half_synthetic_data = half_synthetic_temporal_data.values.reshape(-1, seq_len, temporal_real_data.shape[-1])
    rest_synthetic_data = rest_synthetic_temporal_data.values.reshape(-1, seq_len, temporal_real_data.shape[-1])

    num_patients, seq_len, num_features = train_temporal_data.shape
    train_temporal_data = train_temporal_data.reshape(num_patients, seq_len * num_features)
    num_patients, seq_len, num_features = holdout_temporal_data.shape
    holdout_temporal_data = holdout_temporal_data.reshape(num_patients, seq_len * num_features)
    num_patients, seq_len, num_features = half_synthetic_data.shape
    half_synthetic_data = half_synthetic_data.reshape(num_patients, seq_len * num_features)
    num_patients, seq_len, num_features = rest_synthetic_data.shape
    rest_synthetic_data = rest_synthetic_data.reshape(num_patients, seq_len * num_features)

    exclude_cols = [col for col in static_real_data.columns if 'discharge_location' in col]
    train_static_data = train_static_data.drop(columns=['type'] + exclude_cols)
    holdout_static_data = holdout_static_data.drop(columns=['type'] + exclude_cols)
    half_synthetic_static_data = half_synthetic_static_data.drop(columns=exclude_cols)
    rest_synthetic_static_data = rest_synthetic_static_data.drop(columns=exclude_cols)

    missing_cols = list(half_synthetic_static_data.isnull().sum()[half_synthetic_static_data.isnull().sum() > 0].index)
    for col in missing_cols:
        half_synthetic_static_data[col] = half_synthetic_static_data[col].fillna(
            half_synthetic_static_data[col].mode()[0])
        rest_synthetic_static_data[col] = rest_synthetic_static_data[col].fillna(
            rest_synthetic_static_data[col].mode()[0])

    train_data = np.concatenate([train_static_data, train_temporal_data], axis=1)
    holdout_data = np.concatenate([holdout_static_data, holdout_temporal_data], axis=1)
    half_synthetic_data = np.concatenate([half_synthetic_static_data, half_synthetic_data], axis=1)
    rest_synthetic_data = np.concatenate([rest_synthetic_static_data, rest_synthetic_data], axis=1)

    num_features = len(train_static_data.columns) + len(temporal_real_data.columns)
    subset1 = np.random.choice(np.arange(num_features), size=num_features // 2, replace=False)
    subset2 = np.setdiff1d(np.arange(num_features), subset1)

    cols = [col for col in train_static_data.columns] + [col for col in temporal_real_data.columns]
    subset1_cols = [cols[i] for i in subset1]
    subset2_cols = [cols[i] for i in subset2]

    subset1_indices = []
    subset2_indices = []
    col_names = []
    for col in train_static_data.columns:
        col_names.append(col)

    for t in range(1, seq_len + 1):
        for var in temporal_real_data.columns:
            col_names.append(f"{var}_t{t}")

    subset1_indices = [i for i, col in enumerate(col_names) if col in subset1_cols]
    subset2_indices = [i for i, col in enumerate(col_names) if col in subset2_cols]

    train_data = pd.DataFrame(train_data, columns=col_names)
    holdout_data = pd.DataFrame(holdout_data, columns=col_names)
    half_synthetic_data = pd.DataFrame(half_synthetic_data, columns=col_names)
    rest_synthetic_data = pd.DataFrame(rest_synthetic_data, columns=col_names)

    consistency_real, consistency_synthetic = re_identification_attack(train_data, holdout_data, half_synthetic_data,
                                                                       subset1_indices=subset1_indices,
                                                                       subset2_indices=subset2_indices)

    return consistency_real, consistency_synthetic


def run_attribute_inference_attack(data, data_hat, static_cols, temporal_cols, seq_len, n_timestep):
    static_values = data[static_cols].iloc[np.arange(len(data), step=n_timestep)]
    static_hat_values = data_hat[static_cols].iloc[np.arange(len(data_hat), step=n_timestep)]
    temporal_values = data[temporal_cols]
    temporal_data_hat = data_hat[temporal_cols]

    static_real_data = static_values.reset_index().rename(columns={'index': 'patient_id'})
    static_repeat = pd.DataFrame(np.repeat(static_real_data[['patient_id']].values, seq_len, axis=0),
                                 columns=['patient_id'])
    static_synthetic_data = static_hat_values.reset_index().rename(columns={'index': 'patient_id'})
    static_synthetic_repeat = pd.DataFrame(np.repeat(static_synthetic_data[['patient_id']].values, seq_len, axis=0),
                                           columns=['patient_id'])

    temporal_data = temporal_values[temporal_cols]
    temporal_synthetic_data = temporal_data_hat[temporal_cols]
    temporal_real_data = temporal_data  # pd.concat([time_data, temporal_data], axis=1)
    temporal_synthetic_data = temporal_synthetic_data  # pd.concat([time_data_hat, temporal_synthetic_data], axis=1)
    temporal_real_data = pd.concat([static_repeat, temporal_real_data], axis=1)
    temporal_synthetic_data = pd.concat([static_synthetic_repeat, temporal_synthetic_data], axis=1)

    temporal_real_data = temporal_real_data.fillna(0)
    temporal_synthetic_data = temporal_synthetic_data.fillna(0)

    # 필요한 열만 유지
    temporal_real_data = temporal_real_data[['patient_id'] + temporal_cols]
    temporal_synthetic_data = temporal_synthetic_data[['patient_id'] + temporal_cols]

    real_num_agg = temporal_real_data.groupby('patient_id')[tn_cols].mean()
    synth_num_agg = temporal_synthetic_data.groupby('patient_id')[tn_cols].mean()

    _tc_cols = [col for col in data.columns if '_'.join(col.split('_')[:-1]) in tc_cols]
    real_cat_agg = temporal_real_data.groupby('patient_id')[_tc_cols].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    synth_cat_agg = temporal_synthetic_data.groupby('patient_id')[_tc_cols].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

    real_agg = pd.concat([real_num_agg, real_cat_agg], axis=1).reset_index()
    synth_agg = pd.concat([synth_num_agg, synth_cat_agg], axis=1).reset_index()

    real_data = pd.merge(static_real_data, real_agg, on='patient_id', how='left')
    synthetic_data = pd.merge(static_synthetic_data, synth_agg, on='patient_id', how='left')

    real_data = real_data.drop(columns=['patient_id', 'hours'], errors='ignore')
    synthetic_data = synthetic_data.drop(columns=['patient_id', 'hours'], errors='ignore')

    # fillna
    real_data = real_data.fillna(0)
    synthetic_data = synthetic_data.fillna(0)

    # label encode
    for col in tc_cols + sc_cols:
        combine_cols = [c for c in real_data.columns if col in c]
        if len(combine_cols) == 0:
            continue

        combined = real_data[combine_cols].values.argmax(axis=1)
        real_data[col] = combined
        combined_synth = synthetic_data[combine_cols].values.argmax(axis=1)
        synthetic_data[col] = combined_synth
        real_data = real_data.drop(columns=combine_cols, errors='ignore')

    real_data = real_data.drop(columns=['icu_expire_flag'], errors='ignore')
    synthetic_data = synthetic_data.drop(columns=['icu_expire_flag'], errors='ignore')

    results = {
        'gender': None,
        'marital_status': None,
    }

    # print(' == Gender == ')
    roc_auc_real, roc_auc_synthetic = attribute_inference_attack(
        real_data, synthetic_data, target_col='sex'
    )
    # print(f'Attribute Inference Attack ROC-AUC on Real Data: {np.round(roc_auc_real, 3):.3f}')
    # print(f'Attribute Inference Attack ROC-AUC on Synthetic Data: {np.round(roc_auc_synthetic, 3):.3f}')
    results['gender'] = (roc_auc_real, roc_auc_synthetic)

    # print('== Marital Status == ')
    roc_auc_real, roc_auc_synthetic = attribute_inference_attack(
        real_data, synthetic_data, target_col='marital_status'
    )
    #
    # print(f'Attribute Inference Attack ROC-AUC on Real Data: {np.round(roc_auc_real, 3):.3f}')
    # print(f'Attribute Inference Attack ROC-AUC on Synthetic Data: {np.round(roc_auc_synthetic, 3):.3f}')
    results['marital_status'] = (roc_auc_real, roc_auc_synthetic)

    return results


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
    print(checkpoint)
    eval_file_path = os.path.join(eval_cfg.path.eval_file_path, model_name, checkpoint)
    print(eval_file_path)

    logit_threshold = 0.5
    seq_len = eval_cfg.dataloader.seq_len
    n_timestep = seq_len
    target_col = 'icu_expire_flag'

    static_transformer = dataset.static_transformer
    temporal_transformer = dataset.temporal_transformer

    sc_cols = dataset.sc_cols
    tc_cols = dataset.tc_cols
    sn_cols = dataset.sn_cols
    tn_cols = dataset.tn_cols
    sl_cols = dataset.sl_cols
    tl_cols = dataset.tl_cols

    diagnoses_prefix = eval_cfg.preprocess.icd_code.diagnoses_prefix
    procedure_prefix = eval_cfg.preprocess.icd_code.procedure_prefix
    proc_prefix = eval_cfg.preprocess.proc_prefix

    static_data = pd.read_csv(os.path.join(eval_file_path, 'static_data.csv'))
    temporal_data = pd.read_csv(os.path.join(eval_file_path, 'temporal_data.csv'))
    static_data_hat = pd.read_csv(os.path.join(eval_file_path, 'static_reconstructed_26560.csv'))
    temporal_data_hat = pd.read_csv(os.path.join(eval_file_path, 'temporal_reconstructed_26560.csv'))

    data, fitted = preprocess_data(
        eval_cfg, static_data, temporal_data,
        sn_cols, sc_cols, tn_cols, tc_cols,
        target_col, seq_len, n_timestep,
        exclude_cols=None,  # ['discharge_location'],
        fitted=None,  # fit
        drop_first=False,
        encode=True,
        normalize=False,
        temporal_agg=False
    )

    data_hat = preprocess_data(
        eval_cfg, static_data_hat, temporal_data_hat,
        sn_cols, sc_cols, tn_cols, tc_cols,
        target_col, seq_len, n_timestep,
        exclude_cols=None,
        fitted=fitted,  # transform
        encode=True,
        normalize=False,
        temporal_agg=False
    )

    icd_d_cols = [col for col in static_data.columns if (col.startswith(diagnoses_prefix)) and ('mask' not in col)]
    icd_p_cols = [col for col in static_data.columns if (col.startswith(procedure_prefix)) and ('mask' not in col)]
    proc_cols = [col for col in temporal_data.columns if (col.startswith(proc_prefix)) and ('mask' not in col)]

    static_cols = sn_cols + [col for col in data.columns if
                             '_'.join(col.split('_')[:-1]) in sn_cols + sc_cols] + icd_d_cols + icd_p_cols
    temporal_cols = tn_cols + [col for col in data.columns if '_'.join(col.split('_')[:-1]) in tc_cols] + proc_cols

    save_eval_res_path = os.path.join(eval_file_path, 'evaluation')
    os.makedirs(save_eval_res_path, exist_ok=True)

    # membership inference attack
    print('== Membership Inference Attack ==')
    mem_acc = run_membership_attack(data, data_hat, static_cols, temporal_cols,
                                    seq_len, n_timestep)
    print(f'Membership Inference Attack Accuracy: {np.round(mem_acc, 3):.3f}')

    # re-identification attack
    print('== Re-identification Attack ==')
    consistency_real, consistency_synthetic = run_reidentification_attack(
        data, data_hat, static_cols, temporal_cols,
        seq_len, n_timestep
    )
    print(f'Re-identification Attack Consistency on Real Data: {np.round(consistency_real, 3):.3f}')
    print(f'Re-identification Attack Consistency on Synthetic Data: {np.round(consistency_synthetic, 3):.3f}')

    # attribute inference attack
    print('== Attribute Inference Attack ==')
    attr_results = run_attribute_inference_attack(
        data, data_hat, static_cols, temporal_cols,
        seq_len, n_timestep
    )
    for attr, (roc_auc_real, roc_auc_synthetic) in attr_results.items():
        print(f'Attribute: {attr}')
        print(f'  Attribute Inference Attack ROC-AUC on Real Data: {np.round(roc_auc_real, 3):.3f}')
        print(f'  Attribute Inference Attack ROC-AUC on Synthetic Data: {np.round(roc_auc_synthetic, 3):.3f}')

