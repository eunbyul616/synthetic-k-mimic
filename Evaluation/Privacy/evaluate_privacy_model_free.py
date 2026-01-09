import os
from pathlib import Path
import numpy as np
import pandas as pd

import config_manager

from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset
from Evaluation.Privacy.privacy_metrics import *
from Evaluation.preprocess import *


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

    np.random.seed(eval_cfg.seed)
    n_real = len(static_data)
    n_syn = len(static_data_hat)
    static_data_hat['patient_id'] = np.arange(n_syn)
    temporal_data_hat['patient_id'] = np.repeat(np.arange(n_syn), seq_len)

    sample_patient_ids = np.random.choice(static_data_hat['patient_id'], size=n_real, replace=False)
    static_data_hat = static_data_hat[static_data_hat['patient_id'].isin(sample_patient_ids)].reset_index(drop=True)
    temporal_data_hat = temporal_data_hat[temporal_data_hat['patient_id'].isin(sample_patient_ids)].reset_index(drop=True)
    static_data_hat = static_data_hat.drop(columns=['patient_id'])
    temporal_data_hat = temporal_data_hat.drop(columns=['patient_id'])

    n_timestep = seq_len
    target_col = 'icu_expire_flag'

    data, fitted = preprocess_data(
        eval_cfg, static_data, temporal_data,
        sn_cols, sc_cols, tn_cols, tc_cols,
        target_col, seq_len, n_timestep,
        exclude_cols=None, #['discharge_location'],
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

    static_cols = sn_cols + [col for col in data.columns if '_'.join(col.split('_')[:-1]) in sn_cols + sc_cols] + icd_d_cols + icd_p_cols
    temporal_cols = tn_cols + [col for col in data.columns if '_'.join(col.split('_')[:-1]) in tc_cols] + proc_cols

    save_eval_res_path = os.path.join(eval_file_path, 'evaluation')
    os.makedirs(save_eval_res_path, exist_ok=True)

    static_values = data[static_cols].iloc[np.arange(len(data), step=n_timestep)].values
    static_threshold = calculate_thresholds(data=static_values, iterations=100, quantiles=[0.9, 0.95, 0.99])

    temporal_values = data[temporal_cols].values
    temporal_values = np.mean(temporal_values.reshape((-1, n_timestep, temporal_values.shape[-1])), axis=1)
    temporal_threshold = calculate_thresholds(data=temporal_values, iterations=100, quantiles=[0.9, 0.95, 0.99])

    total_values = data[static_cols + temporal_cols].values
    total_values = np.mean(total_values.reshape((-1, n_timestep, total_values.shape[-1])), axis=1)
    total_threshold = calculate_thresholds(data=total_values, iterations=100, quantiles=[0.9, 0.95, 0.99])

    static_hat_values = data_hat[static_cols].iloc[np.arange(len(data_hat), step=n_timestep)].values
    static_ident_risk = calculate_single_out_risk(static_values, static_hat_values)

    temporal_hat_values = data_hat[temporal_cols].values
    temporal_hat_values = np.mean(temporal_hat_values.reshape((-1, n_timestep, temporal_hat_values.shape[-1])), axis=1)
    temporal_ident_risk = calculate_single_out_risk(temporal_values, temporal_hat_values)

    total_hat_values = data_hat[static_cols + temporal_cols].values
    total_hat_values = np.mean(total_hat_values.reshape((-1, n_timestep, total_hat_values.shape[-1])), axis=1)
    total_ident_risk = calculate_single_out_risk(total_values, total_hat_values)

    print('Static Single Out Risk Thresholds:', static_threshold['single_out_risk_thresholds']['90%'])
    print("Static Single Out Risk:", np.round(static_ident_risk, 3), )
    print('Temporal Single Out Risk Thresholds:', temporal_threshold['single_out_risk_thresholds']['90%'])
    print("Temporal  Single Out Risk:", np.round(temporal_ident_risk, 3))
    print('Total Single Out Risk Thresholds:', total_threshold['single_out_risk_thresholds']['90%'])
    print("Total Single Out Risk:", np.round(total_ident_risk, 3))

    static_inference_risk = calculate_inferential_disclosure_risk(static_values, static_hat_values)
    temporal_inference_risk = calculate_inferential_disclosure_risk(temporal_values, temporal_hat_values)
    total_inference_risk = calculate_inferential_disclosure_risk(total_values, total_hat_values)

    print('Static Inferential Disclosure Risk Thresholds:', static_threshold['inferential_disclosure_risk_thresholds']['90%'])
    print("Static Inferential Disclosure Risk:", np.round(static_inference_risk, 3))
    print('Temporal Inferential Disclosure Risk Thresholds:', temporal_threshold['inferential_disclosure_risk_thresholds']['90%'])
    print("Temporal  Inferential Disclosure Risk:", np.round(temporal_inference_risk, 3))
    print('Total Inferential Disclosure Risk Thresholds:', total_threshold['inferential_disclosure_risk_thresholds']['90%'])
    print("Total Inferential Disclosure Risk:", np.round(total_inference_risk, 3))

    quasi_identifier = ['age', 'sex', 'admission_type']
    print("Calculating Correct Attribution Probability...")
    print('Quasi-Identifier Cols:', quasi_identifier)
    print('Sensitive Cols:', 'icd_d_cols')
    static_data = data[static_cols].iloc[np.arange(len(data), step=n_timestep)].copy()
    static_data_hat = data_hat[static_cols].iloc[np.arange(len(data), step=n_timestep)].copy()
    quasi_identifier_cols = [col for col in static_data.columns if '_'.join(col.split('_')[:-1]) in quasi_identifier]
    quasi_identifier_indices = [static_data.columns.get_loc(col) for col in quasi_identifier_cols]
    sensitive_index = [static_data.columns.get_loc(col) for col in icd_d_cols]

    static_values = static_data.values
    static_hat_values = static_data_hat.values

    cap_thresholds = calculate_cap_thresholds(static_values, iterations=100, quantiles=[0.9, 0.95, 0.99],
                                              quasi_identifier_indices=quasi_identifier_indices, sensitive_index=sensitive_index)
    cap = calculate_correct_attribution_probability(static_values, static_hat_values,
                                                    quasi_identifier_indices=quasi_identifier_indices,
                                                    sensitive_index=sensitive_index)
    print('Thresholds:', cap_thresholds['attribute_disclosure_risk_thresholds']['90%'])
    print("Correct Attribution Probability:", np.round(cap, 3), )

    print('Sensitive Cols:', 'icd_p_cols')
    static_data = data[static_cols].copy()
    static_data_hat = data_hat[static_cols].copy()
    quasi_identifier_cols = [col for col in static_data.columns if
                             '_'.join(col.split('_')[:-1]) in quasi_identifier]
    quasi_identifier_indices = [static_data.columns.get_loc(col) for col in quasi_identifier_cols]
    sensitive_index = [static_data.columns.get_loc(col) for col in icd_p_cols]

    static_values = static_data.values
    static_hat_values = static_data_hat.values
    cap_thresholds = calculate_cap_thresholds(static_values, iterations=100, quantiles=[0.9, 0.95, 0.99],
                                              quasi_identifier_indices=quasi_identifier_indices, sensitive_index=sensitive_index)
    cap = calculate_correct_attribution_probability(static_values, static_hat_values,
                                                    quasi_identifier_indices=quasi_identifier_indices,
                                                    sensitive_index=sensitive_index)
    print('Thresholds:', cap_thresholds['attribute_disclosure_risk_thresholds']['90%'])
    print("Correct Attribution Probability:", np.round(cap, 3))
