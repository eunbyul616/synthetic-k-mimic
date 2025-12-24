import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

from Preprocess.KMIMIC.constants import *
from Preprocess.KMIMIC.utils.utils import preprocess_temporal_data, check_key_type, parse_datetime, convert_col_on_frequency_rank
from Preprocess.KMIMIC.utils.file import read_cohort, load_data, read_events
from Preprocess.utils import check_column_type, convert_type_by_feature_type, save_train_val_test, split_dataset, clip_data_by_timepoints

from Preprocess.KMIMIC.extract_static_features import extract_static_features
from Preprocess.KMIMIC.extract_temporal_features import extract_temporal_features
import config_manager

tqdm.pandas()


def extract_total_features(cfg, save_fname):
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)
    verbose = cfg.preprocess.verbose
    key_cols = cfg.preprocess.keys

    print('Extract Static Features')
    (train_static_data, val_static_data, test_static_data, train_static_mask, val_static_mask, test_static_mask,
     static_feature_type, static_dtypes, static_feature_output_dimensions) = extract_static_features(cfg)

    print('Extract Temporal Features')
    (train_temporal_data, val_temporal_data, test_temporal_data, train_temporal_mask, val_temporal_mask, test_temporal_mask,
     temporal_feature_type, temporal_dtypes, temporal_feature_output_dimensions) = extract_temporal_features(cfg,
                                                                                                             train_static_data,
                                                                                                             val_static_data,
                                                                                                             test_static_data)

    unique_keys = train_temporal_data[key_cols].drop_duplicates()
    train_static_data = pd.merge(unique_keys, train_static_data, on=key_cols, how='inner')
    train_static_mask = pd.merge(unique_keys, train_static_mask, on=key_cols, how='inner')

    unique_keys = val_temporal_data[key_cols].drop_duplicates()
    val_static_data = pd.merge(unique_keys, val_static_data, on=key_cols, how='inner')
    val_static_mask = pd.merge(unique_keys, val_static_mask, on=key_cols, how='inner')

    unique_keys = test_temporal_data[key_cols].drop_duplicates()
    test_static_data = pd.merge(unique_keys, test_static_data, on=key_cols, how='inner')
    test_static_mask = pd.merge(unique_keys, test_static_mask, on=key_cols, how='inner')

    print('Save Static Features')
    save_train_val_test(train_static_data, val_static_data, test_static_data, feature_type=static_feature_type,
                        hdf_key='static', save_path=preprocessed_data_path, save_fname=save_fname,
                        column_dtypes=static_dtypes, feature_output_dimensions=static_feature_output_dimensions)
    save_train_val_test(train_static_mask, val_static_mask, test_static_mask, feature_type=static_feature_type,
                        hdf_key='static_mask', save_path=preprocessed_data_path, save_fname=save_fname,
                        column_dtypes=static_dtypes, feature_output_dimensions=static_feature_output_dimensions)

    print('Save Temporal Features')
    save_train_val_test(train_temporal_data, val_temporal_data, test_temporal_data, feature_type=temporal_feature_type,
                        hdf_key='temporal', save_path=preprocessed_data_path, save_fname=save_fname,
                        column_dtypes=temporal_dtypes, feature_output_dimensions=temporal_feature_output_dimensions)
    save_train_val_test(train_temporal_mask, val_temporal_mask, test_temporal_mask, feature_type=temporal_feature_type,
                        hdf_key='temporal_mask', save_path=preprocessed_data_path, save_fname=save_fname,
                        column_dtypes=temporal_dtypes, feature_output_dimensions=temporal_feature_output_dimensions)


if __name__ == "__main__":
    config_manager.load_config()
    cfg = config_manager.config

    dataset = cfg.preprocess.dataset
    fname = cfg.preprocess.preprocess_fname_suffix
    train_ratio, test_ratio = cfg.preprocess.train_valid_test_split_ratio

    save_fname = f'{dataset}_{fname}_{int(train_ratio*10)}.h5'
    extract_total_features(cfg, save_fname=save_fname)