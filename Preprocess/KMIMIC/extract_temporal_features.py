import os
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from Preprocess.KMIMIC.constants import *

from Preprocess.KMIMIC.utils.utils import (preprocess_temporal_data, check_key_type, parse_datetime,
                                            convert_col_on_frequency_rank, load_variable_ranges)
from Preprocess.KMIMIC.utils.file import read_cohort, load_data, read_events
from Preprocess.utils import (check_column_type, convert_type_by_feature_type, save_train_val_test,
                              split_dataset, clip_data_by_timepoints)
import config_manager

tqdm.pandas()


def transform_categorical_features(data, col, threshold=0.01):
    ratio = data[col].value_counts(normalize=True)
    other_categories = ratio[ratio < threshold].index
    if len(other_categories) > 1:
        categories = ratio[ratio >= threshold].index
        data[col] = data[col].apply(lambda x: x if x in categories else f'{col}_Others')

    return data


def get_base_dataframe(icu_inout) -> pd.DataFrame:
    base_df = icu_inout
    base_df['intime'] = base_df['intime'].apply(parse_datetime)
    base_df['outtime'] = base_df['outtime'].apply(parse_datetime)

    base_df['outtime_hours'] = base_df.apply(lambda x: (x['outtime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    base_df['hours'] = base_df.apply(lambda x: list(range(0, int(x['outtime_hours']) + 1)), axis=1)
    base_df = base_df.drop(columns=['intime', 'outtime', 'outtime_hours'])
    base_df = base_df.explode('hours')

    return base_df


def cutoff_itemid(data, col='itemid', threshold=0.01):
    ratio = data[col].value_counts(normalize=True).map(lambda x: np.round(x, 2))
    return data[data[col].map(ratio) >= threshold]


def preprocess_transfers(icu_inout, transfers, key_cols):
    cohort_subset_keys = list(set(key_cols).intersection(set(icu_inout.columns)))

    subset_keys = list(set(key_cols).intersection(set(transfers.columns)))
    transfers = check_key_type(transfers, subset_keys, 'str')

    subset_keys = list(set(cohort_subset_keys).intersection(set(subset_keys)))
    transfers = transfers.rename(columns={'intime': 'starttime', 'outtime': 'endtime'})
    transfers = icu_inout.merge(transfers, on=subset_keys)
    transfers = transfers[(transfers['starttime'] >= transfers['intime']) &
                          (transfers['starttime'] <= transfers['outtime'])]
    transfers['endtime'] = transfers.apply(lambda x: x['endtime'] if pd.notna(x['endtime']) else x['outtime'], axis=1)

    transfers['starttime'] = transfers['starttime'].apply(parse_datetime)
    transfers['endtime'] = transfers['endtime'].apply(parse_datetime)
    transfers['intime'] = transfers['intime'].apply(parse_datetime)
    transfers['outtime'] = transfers['outtime'].apply(parse_datetime)

    transfers['start_hours'] = transfers.apply(lambda x: (x['starttime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    transfers['end_hours'] = transfers.apply(lambda x: (x['endtime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    transfers['start_hours'] = transfers['start_hours'].astype(HOURS_DTYPE)
    transfers['end_hours'] = transfers['end_hours'].astype(HOURS_DTYPE)
    transfers = transfers.drop(columns=['intime', 'outtime', 'starttime', 'endtime', 'eventtype', 'transfer'])

    transfers['hours'] = transfers.apply(lambda x: list(range(x['start_hours'], x['end_hours'] + 1)), axis=1)
    transfers = transfers.explode('hours').drop(columns=['start_hours', 'end_hours'])
    transfers = transfers[transfers['hours'] >= 0]

    transfers = transfers.groupby(key_cols+['hours']).max().reset_index()

    careunit_columns = [col for col in transfers.columns if col.startswith('careunit_')]
    careunit = transfers[key_cols + ['hours'] + careunit_columns].copy()
    careunit = careunit.set_index(key_cols + ['hours'])
    careunit = careunit.idxmax(axis=1)
    careunit = careunit.reset_index()
    careunit = careunit.rename(columns={0: 'careunit'})

    transfers = transfers.drop(columns=careunit_columns)
    transfers = transfers.merge(careunit, on=key_cols + ['hours'], how='left')

    return transfers


def preprocess_emar(icu_inout, emar, key_cols, ratio_threshold=0.01):
    cohort_subset_keys = list(set(key_cols).intersection(set(icu_inout.columns)))

    subset_keys = list(set(key_cols).intersection(set(emar.columns)))
    emar = check_key_type(emar, subset_keys, 'str')

    subset_keys = list(set(cohort_subset_keys).intersection(set(subset_keys)))
    emar = icu_inout.merge(emar, on=subset_keys)

    # iv fluids
    fluids = ['Plasma solution A',
              'Dextrose 5% & Na K2',
              'Hartmann',
              'Normal saline',
              'Dextrose 5%']
    emar_iv_fluids = emar[emar['medication'].isin(fluids)]
    emar_iv_fluids = cutoff_itemid(emar_iv_fluids, col='medication', threshold=ratio_threshold)
    # iv medications
    emar_medications = emar[~emar['medication'].isin(fluids)]
    emar_medications = cutoff_itemid(emar_medications, col='medication', threshold=ratio_threshold)
    emar = pd.concat([emar_iv_fluids, emar_medications], axis=0)

    emar['charttime'] = emar['charttime'].apply(parse_datetime)
    emar['intime'] = emar['intime'].apply(parse_datetime)

    emar['hours'] = emar.progress_apply(lambda x: (x['charttime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    emar['hours'] = emar['hours'].astype(HOURS_DTYPE)
    emar = emar[emar['hours'] >= 0]
    emar = emar.drop(columns=['intime', 'outtime', 'charttime'])

    # mean
    emar = emar.groupby(key_cols + ['medication', 'hours'])['infusion_rate'].mean().reset_index()
    emar = emar.pivot(index=subset_keys + ['hours'], columns='medication', values='infusion_rate')
    emar = emar.reset_index()
    emar = emar.groupby(key_cols+['hours']).mean().reset_index()

    return emar


def preprocess_inputevents(icu_inout, inputevents, key_cols, ratio_threshold=0.01):
    cohort_subset_keys = list(set(key_cols).intersection(set(icu_inout.columns)))

    subset_keys = list(set(key_cols).intersection(set(inputevents.columns)))
    inputevents = check_key_type(inputevents, subset_keys, 'str')

    subset_keys = list(set(cohort_subset_keys).intersection(set(subset_keys)))
    inputevents = icu_inout.merge(inputevents, on=subset_keys)
    inputevents = cutoff_itemid(inputevents, col='label', threshold=ratio_threshold)

    inputevents['starttime'] = inputevents['starttime'].apply(parse_datetime)
    inputevents['endtime'] = inputevents['endtime'].apply(parse_datetime)
    inputevents['intime'] = inputevents['intime'].apply(parse_datetime)
    inputevents['outtime'] = inputevents['outtime'].apply(parse_datetime)

    inputevents['start_hours'] = inputevents.apply(lambda x: (x['starttime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    inputevents['end_hours'] = inputevents.apply(lambda x: (x['endtime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    inputevents['start_hours'] = inputevents['start_hours'].astype(HOURS_DTYPE)
    inputevents['end_hours'] = inputevents['end_hours'].astype(HOURS_DTYPE)

    inputevents = inputevents.groupby(subset_keys + ['start_hours', 'end_hours', 'label']).agg({
        'amount': 'mean',
        'patientweight': 'mean'
    }).reset_index()

    patient_weight = inputevents[subset_keys + ['start_hours', 'end_hours', 'patientweight']].copy()

    inputevents = inputevents.pivot(index=subset_keys + ['start_hours', 'end_hours'], columns='label', values='amount')
    inputevents = inputevents.reset_index()

    inputevents['hours'] = inputevents.apply(lambda x: list(range(x['start_hours'], x['end_hours'] + 1)), axis=1)
    inputevents = inputevents.explode('hours').drop(columns=['start_hours', 'end_hours'])
    inputevents = inputevents[inputevents['hours'] >= 0]

    patient_weight['hours'] = patient_weight.apply(lambda x: list(range(x['start_hours'], x['end_hours'] + 1)), axis=1)
    patient_weight = patient_weight.explode('hours').drop(columns=['start_hours', 'end_hours'])
    patient_weight = patient_weight[patient_weight['hours'] >= 0]
    patient_weight = patient_weight[patient_weight['patientweight'].notnull()]

    inputevents = inputevents.groupby(key_cols+['hours']).mean().reset_index()
    patient_weight = patient_weight.groupby(key_cols+['hours']).mean().reset_index()

    return inputevents, patient_weight


def preprocess_procedureevents(icu_inout, procedureevents, key_cols, ratio_threshold=0.01, proc_prefix='proc_'):
    cohort_subset_keys = list(set(key_cols).intersection(set(icu_inout.columns)))

    subset_keys = list(set(key_cols).intersection(set(procedureevents.columns)))
    procedureevents = check_key_type(procedureevents, subset_keys, 'str')

    subset_keys = list(set(cohort_subset_keys).intersection(set(subset_keys)))
    procedureevents = icu_inout.merge(procedureevents, on=subset_keys)
    procedureevents = cutoff_itemid(procedureevents, col='label', threshold=ratio_threshold)
    procedureevents = procedureevents.dropna(subset=['starttime', 'endtime'])

    procedureevents['starttime'] = procedureevents['starttime'].apply(parse_datetime)
    procedureevents['endtime'] = procedureevents['endtime'].apply(parse_datetime)
    procedureevents['intime'] = procedureevents['intime'].apply(parse_datetime)
    procedureevents['outtime'] = procedureevents['outtime'].apply(parse_datetime)

    procedureevents['start_hours'] = procedureevents.apply(lambda x: (x['starttime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    procedureevents['end_hours'] = procedureevents.apply(lambda x: (x['endtime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    procedureevents['start_hours'] = procedureevents['start_hours'].astype(HOURS_DTYPE)
    procedureevents['end_hours'] = procedureevents['end_hours'].astype(HOURS_DTYPE)
    procedureevents['flag'] = 1
    # procedureevents = procedureevents.drop_duplicates(subset=subset_keys + ['start_hours', 'end_hours', 'itemid'])
    procedureevents = procedureevents.groupby(key_cols+['label', 'start_hours', 'end_hours']).agg({
        'flag': 'max',
        'patientweight': 'mean'
    }).reset_index()

    patient_weight = procedureevents[subset_keys + ['start_hours', 'end_hours', 'patientweight']].copy()

    procedureevents = procedureevents.pivot(index=subset_keys + ['start_hours', 'end_hours'], columns='label', values='flag')
    procedureevents = procedureevents.fillna(0).reset_index()

    procedureevents['hours'] = procedureevents.apply(lambda x: list(range(x['start_hours'], x['end_hours'] + 1)), axis=1)
    procedureevents = procedureevents.explode('hours').drop(columns=['start_hours', 'end_hours'])
    procedureevents = procedureevents[procedureevents['hours'] >= 0]

    patient_weight['hours'] = patient_weight.apply(lambda x: list(range(x['start_hours'], x['end_hours'] + 1)), axis=1)
    patient_weight = patient_weight.explode('hours').drop(columns=['start_hours', 'end_hours'])
    patient_weight = patient_weight[patient_weight['hours'] >= 0]
    patient_weight = patient_weight[patient_weight['patientweight'].notnull()]

    procedureevents = procedureevents.groupby(key_cols+['hours']).max().reset_index()
    patient_weight = patient_weight.groupby(key_cols+['hours']).mean().reset_index()

    cols = list(set(procedureevents.columns) - set(key_cols + ['hours']))
    # listwise
    procedureevents = procedureevents.rename(columns={col: f'{proc_prefix}{col}' for col in cols})

    return procedureevents, patient_weight


def preprocess_chartevents(icu_inout, events, key_cols, ratio_threshold=0.01):
    cohort_subset_keys = list(set(key_cols).intersection(set(icu_inout.columns)))

    subset_keys = list(set(key_cols).intersection(set(events.columns)))
    events = check_key_type(events, subset_keys, 'str')

    subset_keys = list(set(cohort_subset_keys).intersection(set(subset_keys)))
    events = icu_inout.merge(events, on=subset_keys)
    events = cutoff_itemid(events, col='label', threshold=ratio_threshold)

    events['charttime'] = events['charttime'].progress_apply(parse_datetime)
    events['intime'] = events['intime'].progress_apply(parse_datetime)
    events['hours'] = events.progress_apply(lambda x: (x['charttime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    events['hours'] = events['hours'].astype(HOURS_DTYPE)

    events = events[events['hours'] >= 0]
    events = events.drop(columns=['intime', 'outtime', 'charttime'])
    # events = events.drop_duplicates(subset=subset_keys + ['hours', 'itemid'])

    # Vent mode
    events_vent_mode = events[events['label'] == 'Vent mode']
    events_not_vent_mode = events[events['label'] != 'Vent mode']
    events_not_vent_mode['value'] = events_not_vent_mode['value'].astype(float)
    events_not_vent_mode = events_not_vent_mode.groupby(key_cols+['label', 'hours']).agg({
        'value': 'mean'
    }).reset_index()
    events_vent_mode = events_vent_mode.groupby(key_cols+['label', 'hours']).agg({
        'value': 'first'
    }).reset_index()
    events = pd.concat([events_not_vent_mode, events_vent_mode], axis=0)
    events = events.reset_index(drop=True)
    events = events.pivot(index=subset_keys + ['hours'], columns='label', values='value')
    events = events.reset_index()

    if 'Vent mode' not in events.columns:
        events['Vent mode'] = np.nan

    vent = events[subset_keys + ['hours', 'Vent mode']].copy()
    vent = convert_col_on_frequency_rank(vent, col_name='Vent mode', threshold=ratio_threshold)

    events = events.drop(columns=['Vent mode'])
    events = events.set_index(key_cols + ['hours']).astype(float)
    events = events.groupby(key_cols + ['hours']).mean().reset_index()
    vent = vent.groupby(key_cols + ['hours']).first().reset_index()

    return events, vent


def preprocess_outputevents(icu_inout, events, key_cols, ratio_threshold=0.01):
    cohort_subset_keys = list(set(key_cols).intersection(set(icu_inout.columns)))

    subset_keys = list(set(key_cols).intersection(set(events.columns)))
    events = check_key_type(events, subset_keys, 'str')

    subset_keys = list(set(cohort_subset_keys).intersection(set(subset_keys)))
    events = icu_inout.merge(events, on=subset_keys)
    events = cutoff_itemid(events, col='label', threshold=ratio_threshold)

    events['charttime'] = events['charttime'].progress_apply(parse_datetime)
    events['intime'] = events['intime'].progress_apply(parse_datetime)
    events['hours'] = events.progress_apply(lambda x: (x['charttime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    events['hours'] = events['hours'].astype(HOURS_DTYPE)

    events = events[events['hours'] >= 0]
    events = events.drop(columns=['intime', 'outtime', 'charttime'])
    # events = events.drop_duplicates(subset=subset_keys + ['hours', 'lable'])
    events = events.groupby(key_cols + ['label', 'hours']).agg({
        'value': 'mean'
    }).reset_index()
    events = events.pivot(index=subset_keys + ['hours'], columns='label', values='value')
    events = events.reset_index()

    events = events.set_index(key_cols + ['hours']).astype(float)
    events = events.groupby(key_cols + ['hours']).mean().reset_index()

    return events


def preprocess_labevents(icu_inout, events, key_cols, ratio_threshold=0.01):
    cohort_subset_keys = list(set(key_cols).intersection(set(icu_inout.columns)))

    subset_keys = list(set(key_cols).intersection(set(events.columns)))
    events = check_key_type(events, subset_keys, 'str')

    subset_keys = list(set(cohort_subset_keys).intersection(set(subset_keys)))
    events = icu_inout.merge(events, on=subset_keys)
    events = cutoff_itemid(events, col='label', threshold=ratio_threshold)

    events['charttime'] = events['charttime'].progress_apply(parse_datetime)
    events['intime'] = events['intime'].progress_apply(parse_datetime)
    events['hours'] = events.progress_apply(lambda x: (x['charttime'] - x['intime']).total_seconds() / SECONDS_IN_HOUR, axis=1)
    events['hours'] = events['hours'].astype(HOURS_DTYPE)

    events = events[events['hours'] >= 0]
    events = events.drop(columns=['intime', 'outtime', 'charttime'])
    # events = events.drop_duplicates(subset=subset_keys + ['hours', 'itemid'])
    events = events.groupby(key_cols + ['label', 'hours']).agg({
        'value': 'mean'
    }).reset_index()
    events = events.pivot(index=subset_keys + ['hours'], columns='label', values='value')
    events = events.reset_index()

    events = events.set_index(key_cols + ['hours']).astype(float)
    events = events.groupby(key_cols + ['hours']).mean().reset_index()

    return events


def extract_temporal_features(cfg, train_static_data, val_static_data, test_static_data):
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)
    chunk_size = None
    verbose = cfg.preprocess.verbose

    mapping_path = cfg.path.mapping_path
    variable_ranges = load_variable_ranges(os.path.join(mapping_path, 'variable_ranges.xlsx'))

    key_cols = cfg.preprocess.keys
    time_cols = cfg.preprocess.time_cols
    target_col = cfg.preprocess.target_col
    train_ratio, test_ratio = cfg.preprocess.train_valid_test_split_ratio

    proc_prefix = cfg.preprocess.proc_prefix

    cohort = read_cohort(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.cohort, nrows=debug_num)
    cohort_subset_keys = list(set(key_cols).intersection(set(cohort.columns)))
    cohort = check_key_type(cohort, cohort_subset_keys, 'str')
    icu_inout = cohort[cohort_subset_keys + ['intime', 'outtime']]

    # intime ~ outtime
    transfers = load_data(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.transfers,
                          nrows=debug_num)
    # start ~ end time
    emar = load_data(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.emar,
                     nrows=debug_num)

    # start ~ end time, patient_weight, amount
    inputevents = load_data(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.inputevents,
                            nrows=debug_num)
    # start ~ end time, patient_weight, label
    procedureevents = load_data(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.procedureevents,
                                 nrows=debug_num)

    transfers = preprocess_transfers(icu_inout, transfers, key_cols)
    emar = preprocess_emar(icu_inout, emar, key_cols)

    print("Preprocessing inputevents...")
    inputevents, i_patient_weight = preprocess_inputevents(icu_inout, inputevents, key_cols)

    print("Preprocessing procedureevents...")
    procedureevents, p_patient_weight = preprocess_procedureevents(icu_inout, procedureevents, key_cols,
                                                                   proc_prefix=proc_prefix)
    print("Preprocessing chartevents...")
    chartevents = load_data(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.chartevents,
                            nrows=debug_num)
    chartevents, vent = preprocess_chartevents(icu_inout, chartevents, key_cols)

    print("Preprocessing outputevents...")
    outputevents = load_data(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.outputevents,
                             nrows=debug_num)
    outputevents = preprocess_outputevents(icu_inout, outputevents, key_cols)

    print("Preprocessing labevents...")
    labevents = load_data(preprocessed_data_path, fname=cfg.preprocess.preprocessed_file_name.labevents,
                          nrows=debug_num)
    labevents = preprocess_labevents(icu_inout, labevents, key_cols)

    merge_cols = key_cols + ['hours']
    base_df = get_base_dataframe(icu_inout)

    # remove stays with less than 12 hours
    max_duration = base_df.groupby(key_cols)['hours'].max().reset_index()
    max_duration = max_duration[max_duration['hours'] >= 12]
    base_df = base_df.merge(max_duration[key_cols], on=key_cols, how='inner')

    total_events = base_df.merge(transfers, on=merge_cols, how='left')
    total_events = total_events.merge(emar, on=merge_cols, how='left')
    total_events = total_events.merge(inputevents, on=merge_cols, how='left')
    total_events = total_events.merge(procedureevents, on=merge_cols, how='left')
    total_events = total_events.merge(chartevents, on=merge_cols, how='left')
    total_events = total_events.merge(vent, on=merge_cols, how='left')
    total_events = total_events.merge(outputevents, on=merge_cols, how='left')
    total_events = total_events.merge(labevents, on=merge_cols, how='left')

    i_patient_weight = i_patient_weight[merge_cols + ['patientweight']]
    p_patient_weight = p_patient_weight[merge_cols + ['patientweight']]
    patient_weight = pd.concat([i_patient_weight, p_patient_weight], axis=0)
    patient_weight = patient_weight.groupby(merge_cols).mean().reset_index()
    total_events = total_events.merge(patient_weight, on=merge_cols, how='left')

    dropna = True
    threshold = 7
    feature_types = dict()

    feature_cols = list(set(total_events.columns) - set(key_cols + ['hours']))
    for feat in feature_cols:
        try:
            total_events[feat] = total_events[feat].astype(float)
        except ValueError:
            pass

        values = total_events[feat]
        if pd.api.types.is_numeric_dtype(values):
            if dropna:
                if values.dropna().isin([0, 1]).all():
                    feature_types[feat] = 'Binary'
                else:
                    if (all(values.dropna().apply(lambda x: float(x).is_integer())) and
                            (values.dropna().nunique() <= threshold)):
                        feature_types[feat] = 'Categorical'
                    else:
                        feature_types[feat] = 'Numerical'
            else:
                if values.astype(int).isin([0, 1]).all():
                    feature_types[feat] = 'Binary'
                else:
                    if all(values.apply(lambda x: float(x).is_integer())) and (
                            values.nunique() <= threshold):
                        feature_types[feat] = 'Categorical'
                    else:
                        feature_types[feat] = 'Numerical'

        elif pd.api.types.is_object_dtype(values):
            feature_types[feat] = 'Categorical'
        elif pd.api.types.is_bool_dtype(values):
            feature_types[feat] = 'Binary'
        else:
            feature_types[feat] = 'Other'

    numerical_features = [k for k, v in feature_types.items() if v == 'Numerical']
    categorical_features = [k for k, v in feature_types.items() if v in ('Categorical', 'Binary')]

    time_cols = ['hours']
    tn = total_events[key_cols + time_cols + numerical_features]
    tn[numerical_features] = tn[numerical_features].astype(float)
    tn, tn_mask, tn_feature_type, tn_dtypes = preprocess_temporal_data(data=tn, key_cols=key_cols, time_cols=time_cols,
                                                                       proc_pattern=proc_prefix,
                                                                       outlier=True, variable_ranges=variable_ranges,
                                                                       imputation=True, is_category=False)
    gcs_cols = [col for col in numerical_features if 'Glasgow coma scale' in col]
    tn[gcs_cols] = tn[gcs_cols].astype('int')

    for col in key_cols:
        if col in tn_feature_type.keys():
            del tn_feature_type[col]

    tc = total_events[key_cols + time_cols + categorical_features]
    tc, tc_mask, tc_feature_type, tc_dtypes = preprocess_temporal_data(data=tc, key_cols=key_cols, time_cols=time_cols,
                                                                       proc_pattern=proc_prefix,
                                                                       outlier=False,
                                                                       imputation=True, is_category=True)

    for col in key_cols:
        if col in tc_feature_type.keys():
            del tc_feature_type[col]

    for k, v in tc_feature_type.items():
        if k in key_cols + time_cols:
            continue

        if v == 'Categorical':
            if 'Glasgow coma scale' not in k:
                tc = transform_categorical_features(tc, k)

    temporal_data = pd.merge(tn, tc, on=key_cols + time_cols, how='left')
    temporal_mask = pd.merge(tn_mask, tc_mask, on=key_cols + time_cols, how='left')
    temporal_feature_type = {**tn_feature_type, **tc_feature_type}
    temporal_dtypes = {**tn_dtypes, **tc_dtypes}
    temporal_feature_type[time_cols[0]] = 'Numerical'

    cat_cols = list(set(tc_dtypes.keys()) - set(key_cols) - set(time_cols))
    num_cols = list(tn_dtypes.keys())

    temporal_data = temporal_data[num_cols + cat_cols]
    temporal_mask = temporal_mask[num_cols + cat_cols]

    proc_cols = [col for col in temporal_data.columns if col.startswith(proc_prefix)]
    for col in proc_cols:
        temporal_feature_type[col] = 'Listwise'

    feature_output_dimensions = {}
    for feature, type in temporal_feature_type.items():
        if type in ['Binary', 'Categorical', 'Listwise']:
            if feature in proc_cols:
                if proc_prefix in feature_output_dimensions:
                    feature_output_dimensions[proc_prefix] += 1
                else:
                    feature_output_dimensions[proc_prefix] = 1
            else:
                feature_output_dimensions[feature] = temporal_data[feature].nunique()

        else:
            feature_output_dimensions[feature] = 1

    train_temporal_data, val_temporal_data, test_temporal_data = split_dataset(train_static_data,
                                                                               val_static_data,
                                                                               test_static_data,
                                                                               temporal_data,
                                                                               key_cols=key_cols)
    train_temporal_mask, val_temporal_mask, test_temporal_mask = split_dataset(train_static_data,
                                                                               val_static_data,
                                                                               test_static_data,
                                                                               temporal_mask,
                                                                               key_cols=key_cols)

    train_temporal_data = clip_data_by_timepoints(train_temporal_data,
                                                  timepoints=cfg.preprocess.seq_len,
                                                  padding=cfg.preprocess.pad_flag,
                                                  group_cols=key_cols,
                                                  parallel=cfg.preprocess.parallel)
    val_temporal_data = clip_data_by_timepoints(val_temporal_data,
                                                timepoints=cfg.preprocess.seq_len,
                                                padding=cfg.preprocess.pad_flag,
                                                group_cols=key_cols,
                                                parallel=cfg.preprocess.parallel)
    test_temporal_data = clip_data_by_timepoints(test_temporal_data,
                                                 timepoints=cfg.preprocess.seq_len,
                                                 padding=cfg.preprocess.pad_flag,
                                                 group_cols=key_cols,
                                                 parallel=cfg.preprocess.parallel)
    train_temporal_mask = clip_data_by_timepoints(train_temporal_mask,
                                                  timepoints=cfg.preprocess.seq_len,
                                                  padding=cfg.preprocess.pad_flag,
                                                  group_cols=key_cols,
                                                  parallel=cfg.preprocess.parallel)
    val_temporal_mask = clip_data_by_timepoints(val_temporal_mask,
                                                timepoints=cfg.preprocess.seq_len,
                                                padding=cfg.preprocess.pad_flag,
                                                group_cols=key_cols,
                                                parallel=cfg.preprocess.parallel)
    test_temporal_mask = clip_data_by_timepoints(test_temporal_mask,
                                                 timepoints=cfg.preprocess.seq_len,
                                                 padding=cfg.preprocess.pad_flag,
                                                 group_cols=key_cols,
                                                 parallel=cfg.preprocess.parallel)

    temporal_exclude_cols = [col for col in train_temporal_data.columns if train_temporal_data[col].nunique() == 1]
    if len(temporal_exclude_cols) > 0:
        print(f"Exclude temporal features with only one unique value: {temporal_exclude_cols}")
        train_temporal_data = train_temporal_data.drop(columns=temporal_exclude_cols)
        val_temporal_data = val_temporal_data.drop(columns=temporal_exclude_cols)
        test_temporal_data = test_temporal_data.drop(columns=temporal_exclude_cols)
        train_temporal_mask = train_temporal_mask.drop(columns=temporal_exclude_cols)
        val_temporal_mask = val_temporal_mask.drop(columns=temporal_exclude_cols)
        test_temporal_mask = test_temporal_mask.drop(columns=temporal_exclude_cols)

        for col in temporal_exclude_cols:
            if col in temporal_feature_type.keys():
                del temporal_feature_type[col]
            if col in temporal_dtypes.keys():
                del temporal_dtypes[col]

    return (train_temporal_data, val_temporal_data, test_temporal_data, train_temporal_mask, val_temporal_mask, test_temporal_mask,
    temporal_feature_type, temporal_dtypes, feature_output_dimensions)


if __name__ == "__main__":
    config_manager.load_config()
    cfg = config_manager.config

    dataset = cfg.preprocess.dataset
    save_fname = f'{dataset}_preprocessed.h5'
    extract_temporal_features(cfg, save_fname=save_fname)