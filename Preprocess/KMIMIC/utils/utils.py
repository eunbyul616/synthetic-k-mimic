import os
import pandas as pd
import numpy as np
from typing import List
from time import time
from tqdm import tqdm

from Preprocess.utils import check_column_type, convert_type_by_feature_type, save_train_val_test, get_column_dtype
from Preprocess.KMIMIC.utils.outlier import remove_outlier_on_percentile, remove_outlier_by_limits
from Preprocess.KMIMIC.utils.file import *


def measure_runtime(func):
    """
    Measure the runtime of the function
    Args:
        func: function

    Returns:
        wrapper function
    """
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f'=== Runtime of {func.__name__}: {time() - start:.2f} sec ===\n')
        return result

    return wrapper


def set_cohort(patients: pd.DataFrame,
               admissions: pd.DataFrame,
               stays: pd.DataFrame,
               only_icu_stays: bool=True) -> pd.DataFrame:
    """
    Set the cohort
    Args:
        patients:
        admissions:
        stays:

    Returns:

    """
    data = patients.merge(admissions, on='subject_id')

    if only_icu_stays:
        data = data.merge(stays, on=['subject_id', 'hadm_id'])
    else:
        data = data.merge(stays, on=['subject_id', 'hadm_id'], how='left')

    return data


def remove_icustays_with_transfer(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove icu stays with transfers
    Args:
        data:

    Returns:

    """
    data = data[data['first_careunit'] == data['last_careunit']]
    return data


def remove_multiple_stays_per_admission(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove icu stays with multiple stays per admission
    Args:
        data:

    Returns:

    """
    data = data.sort_values(by='intime')
    data['rank'] = data.groupby('hadm_id').cumcount() + 1
    data = data[data['rank'] == 1]

    return data


def convert_age_unit(data: pd.DataFrame) -> pd.DataFrame:
    def convert_age(row):
        DAYS_IN_YEAR = 365.25
        MONTHS_IN_YEAR = 12

        if row['unit'] == 'days':
            return row['value'] / DAYS_IN_YEAR
        elif row['unit'] == 'months':
            return row['value'] / MONTHS_IN_YEAR
        elif row['unit'] == 'years':
            return row['value']
        else:
            raise ValueError('Invalid unit')

    # check type of age
    if data['anchor_age'].dtype == 'object':
        data['value'] = data['anchor_age'].str.extract('(\d+)').astype(int)
        data['unit'] = data['anchor_age'].str.extract('([a-zA-Z]+)')
        data['anchor_age'] = data.apply(convert_age, axis=1)

    return data


def remove_patients_on_age(data: pd.DataFrame, min_age: int=18, max_age: int=89) -> pd.DataFrame:
    """
    Remove patients with age < min_age or age > max_age
    Args:
        data: cohort data
        min_age: minimum age (default: 18)
        max_age: maximum age (default: 89)

    Returns:
        pd.DataFrame: patients data with age between min_age and max_age
    """
    data = convert_age_unit(data)
    if 'unit' not in data.columns:
        data['unit'] = 'years'

    data = data[(data['anchor_age'] >= min_age) & (data['anchor_age'] <= max_age)]
    data = data.rename(columns={'value': 'age'})
    data = data.drop(columns=['unit'])

    return data


def remove_stays_on_los(data: pd.DataFrame, min_los: int=1, max_los: int=None) -> pd.DataFrame:
    """
    Remove icu stays that exceed the maximum length of stay
    The duration of a patient’s ICU stay is at least 12 hours and less than 10 days.
    ref: Li, J., Cairns, B. J., Li, J., & Zhu, T. (2023). Generating synthetic mixed-type longitudinal electronic health records for artificial intelligent applications. NPJ Digital Medicine, 6(1), 98.

    Args:
        data: cohort data
        min_los: minimum length of stay (default: 1)
        max_los: maximum length of stay (default: 10)

    Returns:
        pd.DataFrame: icu stays data with length of stay less than max_los
    """
    if max_los is not None:
        data = data[(data['los'] >= min_los)&(data['los'] < max_los)]
    else:
        data = data[data['los'] >= min_los]
    return data


def add_inhospital_mortality(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add column 'mortality_inhospital' which means in-hospital mortality
    Args:
        data:

    Returns:

    """
    mortality = data['dod'].notnull() & (data['admittime'] <= data['dod']) & (data['dod'] <= data['dischtime'])
    data['mortality_inhospital'] = mortality.astype(int)

    return data


def partition_by_inhospital_mortality(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Partition the icu stays by in-hospital mortality
    Args:
        data:

    Returns:

    """

    if 'mortality_inhospital' not in data.columns:
        data = add_inhospital_mortality(data)

    pos_cohort = data[data['mortality_inhospital'] == 1]
    neg_cohort = data[data['mortality_inhospital'] == 0]

    return pos_cohort, neg_cohort


def check_key_type(data: pd.DataFrame, keys: list, dtype: str) -> pd.DataFrame:
    """
    Check the type of key columns and convert to the specified type
    """

    for key in keys:
        data[key] = data[key].astype(dtype)

    return data


@measure_runtime
def get_chartevents_on_cohort(cfg,
                              path: str,
                              cohort: pd.DataFrame,
                              folder: str = 'icu',
                              file_name: str = 'chartevents.csv.gz',
                              d_items_file_name: str = 'd_items.csv.gz',
                              compression: str='gzip',
                              nrows: int=None,
                              chunk_size: int=None,
                              verbose: bool=False) -> pd.DataFrame:
    """
    Get data from chartevents table on cohort
    Args:
        path: file path
        cohort: selected cohort data from saved file
        compression: compression type (default: gzip)
        nrows: number of rows to read (default: None)
        chunk_size: chunk size to read (default: None)
        verbose: verbose mode (default: False)

    Returns:
        pd.DataFrame: chartevents data on cohort
    """
    _cohort = cohort[cfg.preprocess.keys+['intime']].drop_duplicates(subset=cfg.preprocess.keys)

    if chunk_size:
        events = []
        d_items = read_d_items_table(path=path, file_name=d_items_file_name, compression=compression, nrows=nrows, linksto='chartevents')
        cols = ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum', 'valueuom']
        path = os.path.join(path, 'icu', 'chartevents.csv')

        if nrows:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression,
                                     parse_dates=['charttime'], chunksize=chunk_size, nrows=nrows):
                chunk = convert_keys_type(chunk, dataset=cfg.preprocess.dataset)
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id', 'stay_id'])
                chunk = chunk.merge(d_items, on='itemid')
                events.append(chunk)
        else:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression,
                                     parse_dates=['charttime'], chunksize=chunk_size):
                chunk = convert_keys_type(chunk, dataset=cfg.preprocess.dataset)
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id', 'stay_id'])
                chunk = chunk.merge(d_items, on='itemid')
                events.append(chunk)
        events = pd.concat(events)

    else:
        chartevents = read_chartevents_table(cfg=cfg, path=path, folder=folder,
                                             file_name=file_name,
                                             d_items_file_name=d_items_file_name,
                                             compression=compression, nrows=nrows)
        keys = list(set(cfg.preprocess.keys).intersection(set(chartevents.columns)))
        # _cohort = check_key_type(_cohort, keys, 'str')
        # chartevents = check_key_type(chartevents, keys, 'str')
        chartevents = check_duplicated_value(chartevents, drop=True)

        subset_keys = list(set(cfg.preprocess.keys).intersection(set(chartevents.columns)))
        _cohort = _cohort[subset_keys].drop_duplicates(subset=subset_keys)
        events = chartevents.merge(_cohort, on=subset_keys)

    # drop rows with missing values in 'value'
    # events = events.dropna(subset=['value'])

    if verbose:
        print('\nTABLE: CHARTEVENTS')
        print('# of Unique Patients:', events['subject_id'].nunique())
        print('# of Unique ICU Stays:', events['stay_id'].nunique())
        print('# of Unique Events:', events['itemid'].nunique())
        print('# of Rows:', events.shape[0])

    return events


@measure_runtime
def get_outputevents_on_cohort(cfg,
                               path: str,
                               cohort: pd.DataFrame,
                               folder: str = 'icu',
                               file_name: str = 'outputevents.csv.gz',
                               d_items_file_name: str = 'd_items.csv.gz',
                               compression: str='gzip',
                               nrows: int=None,
                               chunk_size: int=None,
                               verbose: bool=False) -> pd.DataFrame:
    """
    Get data from outputevents table on cohort
    Args:
        path: file path
        cohort: selected cohort data from saved file
        compression: compression type (default: gzip)
        nrows: number of rows to read (default: None)
        chunk_size: chunk size to read (default: None)
        verbose: verbose mode (default: False)

    Returns:
        pd.DataFrame: outputevents data on cohort
    """
    _cohort = cohort[cfg.preprocess.keys+['intime']].drop_duplicates(subset=cfg.preprocess.keys)

    if chunk_size:
        events = []
        d_items = read_d_items_table(path=path, compression=compression, nrows=nrows, linksto='outputevents')
        cols = ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'valuenum', 'valueuom']
        path = os.path.join(path, 'icu', 'outputevents.csv.gz')

        if nrows:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression,
                                     parse_dates=['charttime'], chunksize=chunk_size, nrows=nrows):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id', 'stay_id'])
                chunk = chunk.merge(d_items, on='itemid')
                events.append(chunk)
        else:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression,
                                     parse_dates=['charttime'], chunksize=chunk_size):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id', 'stay_id'])
                chunk = chunk.merge(d_items, on='itemid')
                events.append(chunk)
        events = pd.concat(events)
    else:
        outputevents = read_outputevents_table(cfg=cfg, path=path, folder=folder,
                                               file_name=file_name,
                                               d_items_file_name=d_items_file_name,
                                               compression=compression, nrows=nrows)
        outputevents = outputevents.rename(columns={'icustay_id': 'stay_id'})

        keys = list(set(cfg.preprocess.keys).intersection(set(outputevents.columns)))
        # _cohort = check_key_type(_cohort, keys, 'str')
        # outputevents = check_key_type(outputevents, keys, 'str')
        outputevents = check_duplicated_value(outputevents, drop=True)

        subset_keys = list(set(cfg.preprocess.keys).intersection(set(outputevents.columns)))
        _cohort = _cohort[subset_keys].drop_duplicates(subset=subset_keys)
        events = outputevents.merge(_cohort, on=subset_keys)

    # drop rows with missing values in 'value'
    # events = events.dropna(subset=['value'])

    if verbose:
        print('\nTABLE: OUTPUTEVENTS')
        print('# of Unique Patients:', events['subject_id'].nunique())
        print('# of Unique ICU Stays:', events['stay_id'].nunique())
        print('# of Unique Events:', events['itemid'].nunique())
        print('# of Rows:', events.shape[0])

    return events


@measure_runtime
def get_procedureevents_on_cohort(cfg,
                                  path: str,
                                  cohort: pd.DataFrame,
                                  folder: str = 'icu',
                                  file_name: str = 'procedureevents.csv.gz',
                                  d_items_file_name: str = 'd_items.csv.gz',
                                  compression: str = 'gzip',
                                  nrows: int = None,
                                  chunk_size: int = None,
                                  verbose: bool = False) -> pd.DataFrame:
    """
    Get data from labevents table on cohort
    Args:
        path: file path
        cohort: selected cohort data from saved file
        compression: compression type (default: gzip)
        nrows: number of rows to read (default: None)
        chunk_size: chunk size to read (default: None)
        verbose: verbose mode (default: False)

    Returns:
        pd.DataFrame: labevents data on cohort
    """
    keys = list(set(cfg.preprocess.keys).intersection(set(cohort.columns)))
    _cohort = cohort[keys + ['intime']].drop_duplicates(subset=keys)

    if chunk_size:
        events = []
        d_labitems = read_d_labitems_table(path=path, compression=compression, nrows=nrows)
        cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valueuom']
        path = os.path.join(path, folder, file_name)
        if nrows:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression,
                                     parse_dates=['charttime'], chunksize=chunk_size, nrows=nrows):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                chunk = chunk.merge(d_labitems, on='itemid')
                events.append(chunk)
        else:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression,
                                     parse_dates=['charttime'], chunksize=chunk_size):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                chunk = chunk.merge(d_labitems, on='itemid')
                events.append(chunk)
        events = pd.concat(events)
    else:
        procedureevents = read_procedureevents_table(cfg=cfg, path=path, folder=folder,
                                                     file_name=file_name,
                                                     d_items_file_name=d_items_file_name,
                                                     compression=compression, nrows=nrows)
        keys = list(set(cfg.preprocess.keys).intersection(set(procedureevents.columns)))
        # _cohort = check_key_type(_cohort, keys, 'str')
        # procedureevents = check_key_type(procedureevents, keys, 'str')
        procedureevents = check_duplicated_value(procedureevents, drop=True)

        subset_keys = list(set(cfg.preprocess.keys).intersection(set(procedureevents.columns)))
        _cohort = _cohort[subset_keys].drop_duplicates(subset=subset_keys)
        procedureevents = procedureevents.merge(_cohort, on=subset_keys)
        events = procedureevents.merge(_cohort, on=keys)

    # drop rows with missing values in 'value'
    # events = events.dropna(subset=['value'])

    if verbose:
        print('\nTABLE: PROCEDUREEVENTS')
        print('# of Unique Patients:', events['subject_id'].nunique())
        print('# of Unique Admissions:', events['hadm_id'].nunique())
        print('# of Unique ICU Stays:', events['stay_id'].nunique())
        print('# of Unique Events:', events['itemid'].nunique())
        print('# of Rows:', events.shape[0])

    return events


@measure_runtime
def get_inputevents_on_cohort(cfg,
                                  path: str,
                                  cohort: pd.DataFrame,
                                  folder: str = 'icu',
                                  file_name: str = 'inputevents.csv.gz',
                                  d_items_file_name: str = 'd_items.csv.gz',
                                  compression: str = 'gzip',
                                  nrows: int = None,
                                  chunk_size: int = None,
                                  verbose: bool = False) -> pd.DataFrame:
    """
    Get data from inputevents table on cohort
    Args:
        path: file path
        cohort: selected cohort data from saved file
        compression: compression type (default: gzip)
        nrows: number of rows to read (default: None)
        chunk_size: chunk size to read (default: None)
        verbose: verbose mode (default: False)

    Returns:
        pd.DataFrame: inputevents data on cohort
    """
    keys = list(set(cfg.preprocess.keys).intersection(set(cohort.columns)))
    _cohort = cohort[keys + ['intime']].drop_duplicates(subset=keys)

    if chunk_size:
        events = []
        d_labitems = read_d_labitems_table(path=path, compression=compression, nrows=nrows)
        cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valueuom']
        path = os.path.join(path, folder, file_name)
        if nrows:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression,
                                     parse_dates=['charttime'], chunksize=chunk_size, nrows=nrows):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                chunk = chunk.merge(d_labitems, on='itemid')
                events.append(chunk)
        else:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression,
                                     parse_dates=['charttime'], chunksize=chunk_size):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                chunk = chunk.merge(d_labitems, on='itemid')
                events.append(chunk)
        events = pd.concat(events)
    else:
        inputevents = read_inputevents_table(cfg=cfg, path=path, folder=folder,
                                                     file_name=file_name,
                                                     d_items_file_name=d_items_file_name,
                                                     compression=compression, nrows=nrows)
        inputevents = inputevents.rename(columns={'icustay_id': 'stay_id'})
        keys = list(set(cfg.preprocess.keys).intersection(set(inputevents.columns)))
        # _cohort = check_key_type(_cohort, keys, 'str')
        # inputevents = check_key_type(inputevents, keys, 'str')
        inputevents = check_duplicated_value(inputevents, drop=True)

        subset_keys = list(set(cfg.preprocess.keys).intersection(set(inputevents.columns)))
        _cohort = _cohort[subset_keys].drop_duplicates(subset=subset_keys)
        events = inputevents.merge(_cohort, on=subset_keys)

    # drop rows with missing values in 'value'
    # events = events.dropna(subset=['value'])

    if verbose:
        print('\nTABLE: INPUTEVENTS')
        print('# of Unique Patients:', events['subject_id'].nunique())
        print('# of Unique Admissions:', events['hadm_id'].nunique())
        print('# of Unique ICU Stays:', events['stay_id'].nunique())
        print('# of Unique Events:', events['itemid'].nunique())
        print('# of Rows:', events.shape[0])

    return events



@measure_runtime
def get_labevents_on_cohort(cfg,
                            path: str,
                            cohort: pd.DataFrame,
                            folder: str = 'hosp',
                            file_name: str = 'labevents.csv.gz',
                            d_labitems_file_name: str = 'd_labitems.csv.gz',
                            compression: str='gzip',
                            nrows: int=None,
                            chunk_size: int=None,
                            verbose: bool=False) -> pd.DataFrame:
    """
    Get data from labevents table on cohort
    Args:
        path: file path
        cohort: selected cohort data from saved file
        compression: compression type (default: gzip)
        nrows: number of rows to read (default: None)
        chunk_size: chunk size to read (default: None)
        verbose: verbose mode (default: False)

    Returns:
        pd.DataFrame: labevents data on cohort
    """
    keys = list(set(cfg.preprocess.keys).intersection(set(cohort.columns)))
    _cohort = cohort[keys+['intime']].drop_duplicates(subset=keys)

    if chunk_size:
        events = []
        d_labitems = read_d_labitems_table(path=path, compression=compression, nrows=nrows)
        cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valueuom']
        path = os.path.join(path, 'hosp', 'labevents.csv.gz')
        if nrows:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression,
                                     parse_dates=['charttime'], chunksize=chunk_size, nrows=nrows):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                chunk = chunk.merge(d_labitems, on='itemid')
                events.append(chunk)
        else:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression,
                                     parse_dates=['charttime'], chunksize=chunk_size):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                chunk = chunk.merge(d_labitems, on='itemid')
                events.append(chunk)
        events = pd.concat(events)
    else:
        labevents = read_labevents_table(cfg=cfg, path=path, folder=folder, file_name=file_name,
                                         d_labitems_file_name=d_labitems_file_name,
                                         compression=compression, nrows=nrows)
        keys = list(set(cfg.preprocess.keys).intersection(set(labevents.columns)))
        # _cohort = check_key_type(_cohort, keys, 'str')
        # labevents = check_key_type(labevents, keys, 'str')
        labevents = check_duplicated_value(labevents, drop=True)

        subset_keys = list(set(cfg.preprocess.keys).intersection(set(labevents.columns)))
        _cohort = _cohort[subset_keys].drop_duplicates(subset=subset_keys)

        events = labevents.merge(_cohort, on=subset_keys)

    # drop rows with missing values in 'value'
    # events = events.dropna(subset=['value'])

    if verbose:
        print('\nTABLE: LABEVENTS')
        print('# of Unique Patients:', events['subject_id'].nunique())
        print('# of Unique Admissions:', events['hadm_id'].nunique())
        print('# of Unique Events:', events['itemid'].nunique())
        print('# of Rows:', events.shape[0])

    return events


@measure_runtime
def get_diagnoses_icd_on_cohort(cfg,
                                path: str,
                                cohort: pd.DataFrame,
                                folder: str='hosp',
                                file_name: str='diagnoses_icd.csv.gz',
                                compression: str='gzip',
                                nrows: int=None,
                                chunk_size: int=None,
                                verbose: bool=False) -> pd.DataFrame:
    """
    Get data from diagnoses_icd table on cohort
    Args:
        path: file path
        cohort: selected cohort data from saved file
        compression: compression type (default: gzip)
        nrows: number of rows to read (default: None)
        chunk_size: chunk size to read (default: None)
        verbose: verbose mode (default: False)

    Returns:
        pd.DataFrame: diagnoses_icd data on cohort
    """
    keys = list(set(cfg.preprocess.keys).intersection(set(cohort.columns)))
    _cohort = cohort[keys].drop_duplicates(subset=keys)

    if chunk_size:
        data = []
        d_icd_diagnoses = read_d_icd_diagnoses_table(cfg=cfg, path=path, compression=compression, nrows=nrows)
        cols = ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version']
        path = os.path.join(path, folder, file_name)

        if nrows:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression, chunksize=chunk_size, nrows=nrows):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                chunk = chunk.merge(d_icd_diagnoses, on='icd_code')
                data.append(chunk)
        else:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression, chunksize=chunk_size):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                chunk = chunk.merge(d_icd_diagnoses, on='icd_code')
                data.append(chunk)
        data = pd.concat(data)
        # drop duplicated rows
        data = check_duplicated_value(data, cols=['subject_id', 'hadm_id', 'seq_num'], drop=True)

    else:
        data = read_diagnoses_icd_table(cfg=cfg, path=path, folder=folder, file_name=file_name, compression=compression, nrows=nrows)

        subset_keys = list(set(cfg.preprocess.keys).intersection(set(data.columns)))
        _cohort = _cohort[subset_keys].drop_duplicates(subset=subset_keys)
        data = data.merge(_cohort, on=subset_keys)

        # drop duplicated rows
        data = check_duplicated_value(data, cols=subset_keys+['seq_num'], drop=True)

    if verbose:
        print('\nTABLE: DIAGNOSES_ICD')
        print('# of Unique Patients:', data['subject_id'].nunique())
        print('# of Unique Admissions:', data['hadm_id'].nunique())
        print('# of Unique Diagnoses:', data['icd_code'].nunique())
        print('# of Rows:', data.shape[0])

    return data


@measure_runtime
def get_procedures_icd_on_cohort(cfg,
                                 path: str,
                                 cohort: pd.DataFrame,
                                 folder: str='hosp',
                                 file_name: str='procedures_icd.csv.gz',
                                 compression: str='gzip',
                                 nrows: int=None,
                                 chunk_size: int=None,
                                 verbose: bool=False) -> pd.DataFrame:
    """
    Get data from diagnoses_icd table on cohort
    Args:
        path: file path
        cohort: selected cohort data from saved file
        compression: compression type (default: gzip)
        nrows: number of rows to read (default: None)
        chunk_size: chunk size to read (default: None)
        verbose: verbose mode (default: False)

    Returns:
        pd.DataFrame: diagnoses_icd data on cohort
    """
    keys = list(set(cfg.preprocess.keys).intersection(set(cohort.columns)))
    _cohort = cohort[keys].drop_duplicates(subset=keys)

    if chunk_size:
        data = []
        cols = ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version']
        path = os.path.join(path, folder, file_name)

        if nrows:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression, chunksize=chunk_size, nrows=nrows):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                data.append(chunk)
        else:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression, chunksize=chunk_size):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                data.append(chunk)
        data = pd.concat(data)
        # drop duplicated rows
        data = check_duplicated_value(data, cols=['subject_id', 'hadm_id', 'seq_num'], drop=True)

    else:
        data = read_procedures_icd_table(cfg=cfg, path=path, folder=folder, file_name=file_name, compression=compression, nrows=nrows)

        subset_keys = list(set(cfg.preprocess.keys).intersection(set(data.columns)))
        _cohort = _cohort[subset_keys].drop_duplicates(subset=subset_keys)
        data = data.merge(_cohort, on=subset_keys)

        # drop duplicated rows
        data = check_duplicated_value(data, cols=subset_keys+['seq_num'], drop=True)

    if verbose:
        print('\nTABLE: PROCEDURES_ICD')
        print('# of Unique Patients:', data['subject_id'].nunique())
        print('# of Unique Admissions:', data['hadm_id'].nunique())
        print('# of Unique Diagnoses:', data['icd_code'].nunique())
        print('# of Rows:', data.shape[0])

    return data


def preprocess_icd_code(diagnoses,
                        path: str,
                        compression: str='infer',
                        nrows: int=None,
                        verbose: bool=False,
                        keep: bool=False,
                        icd_type: str='icd_cm') -> pd.DataFrame:
    """
    Convert icd_code from icd9 to icd10
    Args:
        diagnoses: diagnoses data
        path: file path of icd9 to icd10 mapping table
        fname: file name of icd9 to icd10 mapping table
        compression: compression type (default: infer)
        nrows: number of rows to read (default: None)
        verbose: verbose mode (default: False)
        keep: keep original icd_code (default: False)

    Returns:
        pd.DataFrame: diagnoses data with icd10_code
    """
    assert icd_type in ['icd_cm', 'icd_pcs'], "icd_type must be either 'icd_cm' or 'icd_pcs'"

    # convert icd9 to icd10
    if icd_type == 'icd_cm':
        path = os.path.join(path, 'icd_cm_9_to_10_mapping.csv.gz')
    else:
        path = os.path.join(path, 'icd_pcs_9_to_10_mapping.csv.gz')

    maps = read_icd9_to_icd10_mapping_table(path, compression, nrows)
    diagnoses['icd10_code'] = diagnoses.apply(lambda x:
                                              convert_icd9_to_icd10(x['icd_code'], maps) if x['icd_version'] == 9
                                              else x['icd_code'],
                                              axis=1)

    if verbose:
        original_icd_code = diagnoses[['icd_code', 'icd_version']].groupby(['icd_version']).count()
        converted_icd_code = diagnoses[['icd10_code', 'icd_version']].groupby(['icd_version']).count()
        original_icd_code = original_icd_code.rename(columns={'icd_code': 'count'})
        converted_icd_code = converted_icd_code.rename(columns={'icd10_code': 'count'})

        ratio = (converted_icd_code / original_icd_code)*100

        print('\nICD9 to ICD10 Conversion Ratio')
        print(f'ICD9: {ratio.loc[9]["count"] :.2f}% ({converted_icd_code.loc[9]["count"]}/{original_icd_code.loc[9]["count"]})')
        print(f'ICD10: {ratio.loc[10]["count"] :.2f}% ({converted_icd_code.loc[10]["count"]}/{original_icd_code.loc[10]["count"]})')

    if not keep:
        diagnoses['icd_code'] = diagnoses['icd10_code']
        diagnoses['icd_version'] = 10
        diagnoses = diagnoses.drop(columns=['icd10_code'])

    # drop rows with missing values in 'icd_code'
    diagnoses = check_missing_value(diagnoses, col='icd_code', drop=True)

    return diagnoses


def convert_icd9_to_icd10(icd9: str, maps: pd.DataFrame) -> str or None:
    """
    Convert icd9 code to icd10 code
    Args:
        icd9:

    Returns:

    """
    try:
        return maps[maps['icd_9'] == icd9]['icd_10'].values[0]
    except Exception:
        return


@measure_runtime
def get_transfers_on_cohort(cfg,
                            path: str,
                            cohort: pd.DataFrame,
                            folder: str='hosp',
                            file_name: str='transfers.csv.gz',
                            compression: str='gzip',
                            nrows: int=None,
                            chunk_size: int=None,
                            verbose: bool=False) -> pd.DataFrame:
    """
    Get data from diagnoses_icd table on cohort
    Args:
        path: file path
        cohort: selected cohort data from saved file
        compression: compression type (default: gzip)
        nrows: number of rows to read (default: None)
        chunk_size: chunk size to read (default: None)
        verbose: verbose mode (default: False)

    Returns:
        pd.DataFrame: diagnoses_icd data on cohort
    """
    keys = list(set(cfg.preprocess.keys).intersection(set(cohort.columns)))
    _cohort = cohort[keys].drop_duplicates(subset=keys)

    if chunk_size:
        data = []
        cols = None
        path = os.path.join(path, folder, file_name)

        if nrows:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression, chunksize=chunk_size, nrows=nrows):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                data.append(chunk)
        else:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression, chunksize=chunk_size):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                data.append(chunk)
        data = pd.concat(data)
        data = check_duplicated_value(data, cols=['subject_id', 'hadm_id', 'intime', 'outtime'], drop=True)

    else:
        data = read_transfers_table(cfg=cfg, path=path, folder=folder, file_name=file_name, compression=compression, nrows=nrows)

        subset_keys = list(set(cfg.preprocess.keys).intersection(set(data.columns)))
        _cohort = _cohort[subset_keys].drop_duplicates(subset=subset_keys)
        data = data.merge(_cohort, on=subset_keys)

        # drop duplicated rows
        data = check_duplicated_value(data, cols=subset_keys+['intime', 'outtime'], drop=True)

    if verbose:
        print('\nTABLE: TRANSFERS')
        print('# of Unique Patients:', data['subject_id'].nunique())
        print('# of Unique Admissions:', data['hadm_id'].nunique())
        print('# of Rows:', data.shape[0])

    return data



@measure_runtime
def get_emar_on_cohort(cfg,
                            path: str,
                            cohort: pd.DataFrame,
                            folder: str='hosp',
                            file_name: str='emar.csv.gz',
                            compression: str='gzip',
                            nrows: int=None,
                            chunk_size: int=None,
                            verbose: bool=False) -> pd.DataFrame:
    """
    Get data from diagnoses_icd table on cohort
    Args:
        path: file path
        cohort: selected cohort data from saved file
        compression: compression type (default: gzip)
        nrows: number of rows to read (default: None)
        chunk_size: chunk size to read (default: None)
        verbose: verbose mode (default: False)

    Returns:
        pd.DataFrame: diagnoses_icd data on cohort
    """
    keys = list(set(cfg.preprocess.keys).intersection(set(cohort.columns)))
    _cohort = cohort[keys].drop_duplicates(subset=keys)

    if chunk_size:
        data = []
        cols = None
        path = os.path.join(path, folder, file_name)

        if nrows:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression, chunksize=chunk_size, nrows=nrows):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                data.append(chunk)
        else:
            for chunk in pd.read_csv(path, usecols=cols, compression=compression, chunksize=chunk_size):
                chunk = chunk.merge(_cohort, on=['subject_id', 'hadm_id'])
                data.append(chunk)
        data = pd.concat(data)
        data = check_duplicated_value(data, cols=['subject_id', 'hadm_id', 'charttime'], drop=True)

    else:
        data = read_emar_table(cfg=cfg, path=path, folder=folder, file_name=file_name, compression=compression, nrows=nrows)

        subset_keys = list(set(cfg.preprocess.keys).intersection(set(data.columns)))
        _cohort = _cohort[subset_keys].drop_duplicates(subset=subset_keys)
        data = data.merge(_cohort, on=subset_keys)

        # drop duplicated rows
        data = check_duplicated_value(data, cols=subset_keys+['charttime'], drop=True)

    if verbose:
        print('\nTABLE: EMAR')
        print('# of Unique Patients:', data['subject_id'].nunique())
        print('# of Unique Admissions:', data['hadm_id'].nunique())
        print('# of Rows:', data.shape[0])

    return data


def extract_events_items(events: pd.DataFrame, items: list, keep: bool=False, verbose: bool=False) -> pd.DataFrame:
    """
    Extract events items with label
    Args:
        events:
        items:
        keep: keep lower case label (default: False)

    Returns:

    """

    events = lower_case_column(events, 'label')
    _events = events[events['label_lower'].isin(items)]

    if not keep:
        _events = _events.drop(columns=['label_lower'])

    if verbose:
        print('\nAFTER EXTRACT ITEMS FROM EVENTS TABLE')
        print('# of Unique Patients:', _events['subject_id'].nunique())
        print('# of Unique Events:', _events['itemid'].nunique())
        print('# of Rows:', _events.shape[0])

    return _events


def extract_events_items_with_itemid(events: pd.DataFrame,
                                     itemid_map: dict,
                                     keep: bool=False,
                                     verbose: bool=False) -> pd.DataFrame:
    """
    Extract events items with itemid
    Args:
        events:
        itemid_map:
        keep:
        verbose:

    Returns:

    """
    _events = events[events['itemid'].isin(itemid_map.keys())]

    if keep:
        _events['label_original'] = _events['label']
        _events['label'] = _events['itemid'].apply(lambda x: itemid_map[x])
    else:
        _events['label'] = _events['itemid'].apply(lambda x: itemid_map[x])

    if verbose:
        print('\nAFTER EXTRACT ITEMS FROM EVENTS TABLE (WITH ITEMID)')
        print('# of Unique Patients:', _events['subject_id'].nunique())
        print('# of Unique Events:', _events['itemid'].nunique())
        print('# of Rows:', _events.shape[0])

    return _events


def lower_case_column(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Lower case columns
    Args:
        data: data
        col_name: column name to lower case

    Returns:
        pd.DataFrame: data with lower case column name
    """
    data[f'{col_name}_lower'] = data[col_name].map(lambda x: str(x).lower().strip())

    return data


def add_hours_elpased_to_events(data: pd.DataFrame, threshold_time_col: str='chart_time') -> pd.DataFrame:
    """
    Add hours elapsed
    Args:
        data: data

    Returns:
        pd.DataFrame: data with hours elapsed
    """
    data['hours'] = (data[threshold_time_col] - data['intime']).dt.total_seconds() / 60 / 60

    return data


def pivot_icd(data: pd.DataFrame, key_cols: List[str], col_name: str='icd_code', fillna: bool=True, value: int=0, icd_code_col_pattern: str='icd_') -> pd.DataFrame:
    """
    Pivot diagnoses data
    Args:
        data: diagnoses data
        fillna: fill missing values (default: True)
        value: value to fill missing values (default: 0)
        icd_code_col_pattern: icd code column pattern (default: 'icd_')

    Returns:
        pd.DataFrame: pivoted diagnoses data
    """
    data['flag'] = 1
    keys = list(set(key_cols).intersection(set(data.columns)))
    data = data.pivot(index=keys+['seq_num'], columns=col_name, values='flag')
    data = data.groupby(keys).max()
    if fillna:
        data = data.fillna(value)

    if icd_code_col_pattern:
        data.columns = [f'{icd_code_col_pattern}{col}' for col in data.columns]

    data = data.reset_index()

    return data


def one_hot_encode(data: pd.DataFrame, col_name: str, prefix: str='', drop: bool=False) -> pd.DataFrame:
    """
    One-hot encode categorical column
    Args:
        data: data
        col_name: column name to one-hot encode
        prefix: prefix for new columns (default: '')

    Returns:
        pd.DataFrame: data with one-hot encoded columns
    """
    one_hot = pd.get_dummies(data[col_name], prefix=prefix, drop_first=drop)
    one_hot = one_hot.astype(int)
    data = pd.concat([data, one_hot], axis=1)
    data = data.drop(columns=[col_name])

    return data


def check_duplicated_value(data: pd.DataFrame, cols: list=None, drop: bool=False) -> pd.DataFrame:
    """
    Check duplicated value
    Args:
        data: data
        cols: columns to check duplicated rows (default: None)
        drop: drop duplicated rows (default: False)

    Returns:
        pd.DataFrame: data with or without duplicated rows
    """
    if cols:
        duplicated = data.duplicated(subset=cols)
        if duplicated.sum() > 0:
            print(f'\n# of Duplicated Rows (subset {", ".join(cols)}): {duplicated.sum()}')
            if drop:
                data = data.drop_duplicates(subset=cols)

    else:
        duplicated = data.duplicated()
        if duplicated.sum() > 0:
            print(f'\n# of Duplicated Rows: {duplicated.sum()}')
            if drop:
                data = data.drop_duplicates()

    return data


def check_missing_value(data: pd.DataFrame, col: str, drop: bool=False) -> pd.DataFrame:
    """
    Check missing value
    Args:
        data: data
        col: column name to check missing values
        drop: drop rows with missing values (default: False)

    Returns:
        pd.DataFrame: data with or without missing values
    """
    missing = data[col].isnull()
    if missing.sum() > 0:
        print(f'\n# of Missing Values ({col}): {missing.sum()}')
        if drop:
            data = data.dropna(subset=[col])

    return data


# def convert_col_on_frequency_rank(data: pd.DataFrame, col_name='icd_code', ratio: float=0.01, keep: bool=False) -> pd.DataFrame:
#     """
#     Convert column that ranks below num_items to 'others'
#
#     Args:
#         data: diagnoses data
#         num_items: number of items to select (default: 2000)
#         keep: keep original icd code (default: False)
#
#     Returns:
#         pd.DataFrame: diagnoses data with converted icd code
#     """
#     value_count = data[col_name].value_counts()
#     value_count = value_count.sort_values(ascending=False)
#     threshold = int(len(data) * ratio)
#     value_count = value_count[value_count > threshold].index
#
#     if keep:
#         data[f'{col_name}_new'] = data[col_name].apply(lambda x: x if x in value_count else 'Others')
#     else:
#         data[col_name] = data[col_name].apply(lambda x: x if x in value_count else 'Others')
#
#     return data


def convert_col_on_frequency_rank(data: pd.DataFrame, col_name='icd_code', threshold: float=0.01, keep: bool=False) -> pd.DataFrame:
    """
    Convert column that ranks below num_items to 'others'

    Args:
        data: diagnoses data
        num_items: number of items to select (default: 2000)
        keep: keep original icd code (default: False)

    Returns:
        pd.DataFrame: diagnoses data with converted icd code
    """
    ratio = data[col_name].value_counts(normalize=True)
    other_categories = ratio[ratio < threshold].index
    if len(other_categories) > 1:
        categories = ratio[ratio >= threshold].index

        if keep:
            data[f'{col_name}_new'] = data[col_name].apply(lambda x: x if pd.isna(x) or (x in categories) else 'Others')
        else:
            data[col_name] = data[col_name].apply(lambda x: x if pd.isna(x) or (x in categories) else 'Others')

    return data


def merge_cohort_diagnoses(cohort: pd.DataFrame,
                           diagnoses: pd.DataFrame,
                           key_cols_pattern: str='_id',
                           icd_code_col_pattern: str='icd_',
                           use_cols: list=None,
                           drop_duplicate: bool=True) -> pd.DataFrame:
    """
    Merge cohort data with diagnoses data. Get static information.
    Args:
        cohort: cohort data
        diagnoses: diagnoses data
        use_cols: columns to use
        drop_duplicate: drop duplicated rows on subject_id and hadm_id.
        If False, keep the rows have the same subject_id and hadm_id and different stay_id. (default: True)

    Returns:
        pd.DataFrame: static information
    """
    if drop_duplicate:
        _cohort = check_duplicated_value(cohort, cols=['subject_id', 'hadm_id'], drop=True)
    else:
        _cohort = cohort

    data = _cohort.merge(diagnoses, on=['subject_id', 'hadm_id'])

    if use_cols:
        key_cols = [col for col in data.columns if re.search(key_cols_pattern, col)]
        icd_code_cols = [col for col in data.columns if re.search(icd_code_col_pattern, col)]
        data = data[key_cols + use_cols + icd_code_cols]

    return data


def filter_event_items_by_ratio(data: pd.DataFrame, ratio_threshold: float=0.01):
    item_ratio = data.groupby('itemid')['itemid'].count() / data.shape[0]
    item_ratio = item_ratio[item_ratio >= ratio_threshold]

    # filter events by item ratio
    data = data[data['itemid'].isin(item_ratio.index)]

    return data


def sample_timepoints(data: pd.DataFrame,
                      clip_time_threshold: pd.DataFrame,
                      col: str,
                      start_time_col: str,
                      end_time_col: str,
                      timepoints: int,
                      key_cols: List[str]) -> pd.DataFrame:
    keys = list(set(key_cols).intersection(set(data.columns)))
    keys = list(set(keys).intersection(set(clip_time_threshold.columns)))
    merged_data = data.merge(clip_time_threshold, on=keys)
    filtered_data = merged_data[(merged_data[col] >= merged_data[start_time_col]) &
                                (merged_data[col] < merged_data[end_time_col])]
    filtered_data = filtered_data.drop_duplicates(subset=key_cols+[col])

    filtered_data['rank'] = filtered_data.groupby(key_cols)[col].rank(method='first')
    sampled_data = filtered_data[filtered_data['rank'] <= timepoints].drop(columns='rank')

    return sampled_data[key_cols+[col]]


def preprocess_static_data(data: pd.DataFrame,
                           key_cols: List[str],
                           diagnoses_icd_pattern: str='icd_d',
                           procedures_icd_pattern: str='icd_p',
                           exclude_cols: List[str]=None,
                           ) -> pd.DataFrame:
    d_icd_cols = [col for col in data.columns if diagnoses_icd_pattern in col]
    data[f'{diagnoses_icd_pattern}Others'] = data[f'{diagnoses_icd_pattern}Others'].fillna(1)
    data[d_icd_cols] = data[d_icd_cols].fillna(0)

    p_icd_cols = [col for col in data.columns if procedures_icd_pattern in col]
    data[f'{procedures_icd_pattern}Others'] = data[f'{procedures_icd_pattern}Others'].fillna(1)
    data[p_icd_cols] = data[p_icd_cols].fillna(0)

    original_dtypes = get_column_dtype(data)
    key_cols = list(set(key_cols).intersection(set(data.columns)))

    if exclude_cols is not None:
        data = data.drop(columns=exclude_cols)

    feature_type = check_column_type(data)
    mask = data.set_index(key_cols).map(lambda x: 0 if pd.isnull(x) else 1).reset_index()

    return data, mask, feature_type, original_dtypes


def preprocess_temporal_data(data: pd.DataFrame,
                             key_cols: List[str],
                             time_cols: List[str],
                             proc_pattern: str = 'proc_',
                             exclude_cols: List[str]=None,
                             outlier: bool=True,
                             variable_ranges: dict=None,
                             imputation: bool=True,
                             is_category: bool=False) -> (pd.DataFrame, pd.DataFrame, dict, dict):
    proc_cols = [col for col in data.columns if proc_pattern in col]
    if len(proc_cols) > 0:
        data[f'{proc_pattern}Others'] = data[f'{proc_pattern}Others'].fillna(1)
        data[proc_cols] = data[proc_cols].fillna(0)

    original_dtypes = get_column_dtype(data)
    key_cols = list(set(key_cols).intersection(set(data.columns)))

    if exclude_cols is not None:
        data = data.drop(columns=exclude_cols)

    feature_type = check_column_type(data)
    cols = list(set(data.columns).difference(set(key_cols + time_cols)))
    if outlier:
        outlier_bound = dict()

        for col in cols:
            if variable_ranges and col in variable_ranges:
                lower_limit = variable_ranges[col]['lower_limit']
                upper_limit = variable_ranges[col]['upper_limit']
                include_lower = bool(variable_ranges[col]['include_lower'])
                include_upper = bool(variable_ranges[col]['include_upper'])
                data[col], outlier_bound[col] = remove_outlier_by_limits(data[col], lower_limit, upper_limit, include_lower, include_upper)

    mask = data.set_index(key_cols+time_cols).map(lambda x: 0 if pd.isnull(x) else 1).reset_index()

    if imputation:
        if is_category:
            for col in tqdm(cols, desc='Imputing missing values (category)'):
                data[col] = impute_missing_values_category(data, key_cols=['subject_id'], col=col, method='mode')
        else:
            for col in tqdm(cols, desc='Imputing missing values (numerical)'):
                data[col] = impute_missing_values(data, key_cols=['subject_id'], col=col, method='mean')

    return data, mask, feature_type, original_dtypes


def impute_missing_values(data: pd.DataFrame,
                          key_cols: list,
                          col: str,
                          method: str='median') -> pd.Series:
    """"
    Impute missing values

    Args:
        data: data
        key_cols: key columns
        col: column to impute
        method: imputation method

    Returns:
        pd.Series: imputed data
    """
    g = data.groupby(key_cols, sort=False, group_keys=False)[col]
    s = g.apply(lambda x: x.astype(float)
                          .interpolate(method='linear', limit_direction='both')
                          .ffill()
                          .bfill())

    if method == 'median':
        stat = g.transform('median')
        overall = data[col].median()
    elif method == 'mean':
        stat = g.transform('mean')
        overall = data[col].mean()
    else:
        raise ValueError('Invalid method')

    s = s.fillna(stat).fillna(overall)

    assert s.index.equals(data.index)

    return s


def impute_missing_values_category(data: pd.DataFrame,
                                   key_cols: list,
                                   col: str,
                                   method: str='mode') -> pd.Series:
    """"
    Impute missing values

    Args:
        data: data
        key_cols: key columns
        col: column to impute
        method: imputation method

    Returns:
        pd.Series: imputed data
    """
    g = data.groupby(key_cols, sort=False, group_keys=False)[col]

    try:
        s = g.ffill().bfill()
    except Exception:
        s = g.apply(lambda x: x.ffill().bfill())

    if method != 'mode':
        raise ValueError('Invalid method')

    def _mode1(x):
        vc = x.value_counts(dropna=True)
        return vc.idxmax() if len(vc) else np.nan

    grp_mode = g.transform(_mode1)
    s = s.fillna(grp_mode)

    vc_all = data[col].value_counts(dropna=True)
    overall = vc_all.idxmax() if len(vc_all) else np.nan
    s = s.fillna(overall)

    assert s.index.equals(data.index)

    return s


def is_number(s):
    try:
        float(s)
        return True

    except ValueError:
        return False


def load_label_mapping_table(path: str) -> dict:
    label_map = pd.read_excel(path)
    key = label_map.columns[0]
    value = label_map.columns[3]
    label_map = label_map.set_index(key)[value].to_dict()

    return label_map


def load_variable_ranges(path: str) -> dict:
    variable_ranges = pd.read_excel(path)
    variable_ranges_map = {row['label_modified']: {
        'lower_limit': row['lower_limit'] if not pd.isna(row['lower_limit']) else None,
        'upper_limit': row['upper_limit'] if not pd.isna(row['upper_limit']) else None,
        'include_lower': row['include_lower'] if not pd.isna(row['include_lower']) else None,
        'include_upper': row['include_upper'] if not pd.isna(row['include_upper']) else None
    } for _, row in variable_ranges.iterrows()}

    return variable_ranges_map


def parse_datetime(x):
    from datetime import datetime

    if isinstance(x, str):
        try:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            try:
                return datetime.strptime(x, '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return x




if __name__ == '__main__':
    import config_manager
    config_manager.load_config()
    cfg = config_manager.config

    mapping_path = cfg.path.mapping_path

    load_label_mapping_table(os.path.join(mapping_path, 'emar_count.xlsx'))

    breakpoint()