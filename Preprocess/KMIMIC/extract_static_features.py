import os
import pandas as pd
import numpy as np
from datetime import datetime

from Preprocess.KMIMIC.utils.utils import preprocess_temporal_data, preprocess_static_data, check_key_type, sample_timepoints, one_hot_encode
from Preprocess.KMIMIC.utils.file import read_cohort, load_data, read_events
from Preprocess.utils import check_column_type, convert_type_by_feature_type, save_train_val_test, split_dataset, clip_data_by_timepoints
from Utils.dataset import stratified_shuffle_split
import config_manager


def transform_categorical_features(data, col, threshold=0.01):
    ratio = data[col].value_counts(normalize=True)
    other_categories = ratio[ratio < threshold].index
    if len(other_categories) > 1:
        categories = ratio[ratio >= threshold].index
        data[col] = data[col].apply(lambda x: x if x in categories else 'Others')

    return data

def extract_static_features(cfg):
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)
    chunk_size = None
    verbose = cfg.preprocess.verbose

    key_cols = cfg.preprocess.keys
    target_col = cfg.preprocess.target_col
    train_ratio, test_ratio = cfg.preprocess.train_valid_test_split_ratio
    icd_d_prefix = cfg.preprocess.icd_code.diagnoses_prefix
    icd_p_prefix = cfg.preprocess.icd_code.procedure_prefix

    cohort = read_cohort(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.cohort, nrows=debug_num)
    diagnoses = load_data(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.diagnoses_icd,
                          nrows=debug_num)
    procedures = load_data(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.procedures_icd,
                           nrows=debug_num)

    cohort_subset_keys = list(set(key_cols).intersection(set(cohort.columns)))
    cohort = check_key_type(cohort, cohort_subset_keys, 'str')

    diagnoses_subset_keys = list(set(key_cols).intersection(set(diagnoses.columns)))
    diagnoses = check_key_type(diagnoses, diagnoses_subset_keys, 'str')

    procedures_subset_keys = list(set(key_cols).intersection(set(procedures.columns)))
    procedures = check_key_type(procedures, procedures_subset_keys, 'str')

    subset_keys = list(set(cohort_subset_keys).intersection(set(diagnoses_subset_keys)))
    static_data = cohort.merge(diagnoses, on=subset_keys, how='left')

    subset_keys = list(set(subset_keys).intersection(set(procedures_subset_keys)))
    static_data = static_data.merge(procedures, on=subset_keys, how='left')

    target_df = static_data[key_cols + [target_col]].copy()

    static_data, static_mask, static_feature_type, static_dtypes = preprocess_static_data(
        data=static_data,
        key_cols=key_cols,
        diagnoses_icd_pattern=icd_d_prefix,
        procedures_icd_pattern=icd_p_prefix,
        exclude_cols=['anchor_age',
                      'admittime',
                      'dischtime',
                      'edregtime',
                      'edouttime',
                      'deathtime',
                      'intime',
                      'outtime'])
    static_feature_type = check_column_type(static_data.set_index(key_cols))
    static_data = convert_type_by_feature_type(static_data, static_feature_type)

    categorical_features = [k for k, v in static_feature_type.items() if v == 'Categorical']
    for col in categorical_features:
        static_data = transform_categorical_features(static_data, col)

    # icd_d and icd_p is listwise
    icd_d_cols = [col for col in static_data.columns if col.startswith(icd_d_prefix)]
    icd_p_cols = [col for col in static_data.columns if col.startswith(icd_p_prefix)]
    for col in icd_d_cols + icd_p_cols:
        static_feature_type[col] = 'Listwise'

    feature_output_dimensions = {}
    for feature, type in static_feature_type.items():
        if type in ['Binary', 'Categorical', 'Listwise']:
            if feature in icd_d_cols:
                if icd_d_prefix in feature_output_dimensions:
                    feature_output_dimensions[icd_d_prefix] += 1
                else:
                    feature_output_dimensions[icd_d_prefix] = 1
            elif feature in icd_p_cols:
                if icd_p_prefix in feature_output_dimensions:
                    feature_output_dimensions[icd_p_prefix] += 1
                else:
                    feature_output_dimensions[icd_p_prefix] = 1
            else:
                feature_output_dimensions[feature] = static_data[feature].nunique()

        else:
            feature_output_dimensions[feature] = 1

    train_indices, val_indices, test_indices = stratified_shuffle_split(target_df[target_col],
                                                                        patient_ids=target_df[cfg.preprocess.patient_id],
                                                                        train_ratio=train_ratio,
                                                                        test_ratio=test_ratio)

    train_static_data = static_data.iloc[train_indices]
    val_static_data = static_data.iloc[val_indices]
    test_static_data = static_data.iloc[test_indices]
    train_static_mask = static_mask.iloc[train_indices]
    val_static_mask = static_mask.iloc[val_indices]
    test_static_mask = static_mask.iloc[test_indices]

    static_exclude_cols = [col for col in train_static_data.columns if train_static_data[col].nunique() == 1]
    if len(static_exclude_cols) > 0:
        print(f'Excluding static columns with only one unique value: {static_exclude_cols}')
        train_static_data = train_static_data.drop(columns=static_exclude_cols)
        val_static_data = val_static_data.drop(columns=static_exclude_cols)
        test_static_data = test_static_data.drop(columns=static_exclude_cols)
        train_static_mask = train_static_mask.drop(columns=static_exclude_cols)
        val_static_mask = val_static_mask.drop(columns=static_exclude_cols)
        test_static_mask = test_static_mask.drop(columns=static_exclude_cols)
        for col in static_exclude_cols:
            static_feature_type.pop(col, None)
            static_dtypes.pop(col, None)

    return (train_static_data, val_static_data, test_static_data, train_static_mask, val_static_mask, test_static_mask,
            static_feature_type, static_dtypes, feature_output_dimensions)


if __name__ == "__main__":
    config_manager.load_config()
    cfg = config_manager.config
    dataset = cfg.preprocess.dataset
    save_fname = f'{dataset}_preprocessed.h5'
    extract_static_features(cfg)