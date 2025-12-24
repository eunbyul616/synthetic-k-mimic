import os
import pandas as pd
from typing import List
import h5py


def convert_keys_type(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == 'MIMIC_IV':
        from Preprocess.MIMIC.constants import SUBJECT_ID_TYPE, HADM_ID_TYPE, STAY_ID_TYPE
    else:
        from Preprocess.K_MIMIC.constants import SUBJECT_ID_TYPE, HADM_ID_TYPE, STAY_ID_TYPE

    if 'subject_id' in data.columns:
        data['subject_id'] = data['subject_id'].astype(SUBJECT_ID_TYPE)

    if 'hadm_id' in data.columns:
        data['hadm_id'] = data['hadm_id'].astype(HADM_ID_TYPE)


    if 'stay_id' in data.columns:
        if STAY_ID_TYPE == 'float':
            data['stay_id'] = data['stay_id'].astype(STAY_ID_TYPE)
        else:
            data['stay_id'] = (
                data['stay_id']
                .fillna('nan')
                .astype(str)
                .str.replace(r'\.0$', '', regex=True)
                .replace('nan', None)
            )
    if 'icustay_id' in data.columns:
        if STAY_ID_TYPE == 'float':
            data['icustay_id'] = data['icustay_id'].astype(STAY_ID_TYPE)
        else:
            data['icustay_id'] =(
                data['icustay_id']
                .fillna('nan')
                .astype(str)
                .str.replace(r'\.0$', '', regex=True)
                .replace('nan', None)
            )

    return data


def read_file(path, usecols: List[str]=None, compression: str=None, nrows: int=None, parse_dates: list=None, dataset=None) -> pd.DataFrame:
    file_name = os.path.basename(path)
    if nrows:
        if file_name.endswith('.csv.gz'):
            data = pd.read_csv(path, usecols=usecols, compression=compression, nrows=nrows, parse_dates=parse_dates)
        elif file_name.endswith('.csv'):
            data = pd.read_csv(path, usecols=usecols, nrows=nrows, parse_dates=parse_dates)
        elif file_name.endswith('.parquet'):
            data = pd.read_parquet(path, columns=usecols)
            data = data.head(nrows)
        else:
            raise ValueError('file_name should end with .csv.gz or .parquet')
    else:
        if file_name.endswith('.csv.gz'):
            data = pd.read_csv(path, usecols=usecols, compression=compression, parse_dates=parse_dates)
        elif file_name.endswith('.csv'):
            data = pd.read_csv(path, usecols=usecols, parse_dates=parse_dates)
        elif file_name.endswith('.parquet'):
            data = pd.read_parquet(path, columns=usecols)
        else:
            raise ValueError('file_name should end with .csv.gz or .csv or .parquet')

    data = convert_keys_type(data, dataset=dataset)

    return data


def read_patients_table(cfg,
                        path: str,
                        folder: str='hosp',
                        file_name: str='patients.csv.gz',
                        compression: str='gzip',
                        nrows: int=None,
                        included_fname: str='patients_included.txt') -> pd.DataFrame:
    """
    Read the patients table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from patients table
    """

    if included_fname is not None:
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['subject_id', 'gender', 'anchor_age', 'dod']
    path = os.path.join(path, folder, file_name)

    # check file_name whether end with .csv.gz
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, parse_dates=['dod'], dataset=cfg.preprocess.dataset)

    return data


def read_admissions_table(cfg,
                          path: str,
                          folder: str='hosp',
                          file_name: str='admissions.csv.gz',
                          compression: str='gzip', nrows: int=None,
                          included_fname: str='admissions_included.txt') -> pd.DataFrame:
    """
    Read the admissions table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from admissions table
    """
    if included_fname is not None:
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime',
                'marital_status', 'ethnicity', 'insurance', 'hospital_expire_flag']

    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, parse_dates=['admittime', 'dischtime'], dataset=cfg.preprocess.dataset)

    return data


def read_icustays_table(cfg,
                        path: str,
                        folder: str='icu',
                        file_name: str='icustays.csv.gz',
                        compression: str='gzip',
                        nrows: int=None,
                        included_fname: str='icustays_included.txt') -> pd.DataFrame:
    """
    Read the ICU stays table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from icustays table
    """
    if included_fname is not None:
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'first_careunit', 'last_careunit', 'los']
    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, parse_dates=['intime', 'outtime'], dataset=cfg.preprocess.dataset)

    return data


def read_d_icd_diagnoses_table(cfg,
                               path: str,
                               folder: str='hosp',
                               file_name: str='d_icd_diagnoses.csv.gz',
                               compression: str = 'gzip',
                               nrows: int = None,
                               included_fname: str=None) -> pd.DataFrame:
    """
    Read the d_icd_diagnoses table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from d_icd_diagnoses table
    """
    if included_fname is not None:
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['icd_code', 'long_title']
    path = os.path.join(path, folder, file_name)

    if nrows:
        if file_name.endswith('.csv.gz'):
            data = pd.read_csv(path, usecols=cols, compression=compression, nrows=nrows)
        elif file_name.endswith('.parquet'):
            data = pd.read_parquet(path, columns=cols, nrows=nrows)
        else:
            raise ValueError('file_name should end with .csv.gz or .parquet')
    else:
        if file_name.endswith('.csv.gz'):
            data = pd.read_csv(path, usecols=cols, compression=compression)
        elif file_name.endswith('.parquet'):
            data = pd.read_parquet(path, columns=cols)
        else:
            raise ValueError('file_name should end with .csv.gz or .parquet')

    return data


def read_diagnoses_icd_table(cfg,
                             path: str,
                             folder: str='hosp',
                             file_name: str='diagnoses_icd.csv.gz',
                             compression: str = 'gzip',
                             nrows: int = None,
                             included_fname: str='diagnoses_icd_included.txt') -> pd.DataFrame:
    """
    Read the diagnoses_icd table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from diagnoses_icd table and d_icd_diagnoses table. Two tables are merged on 'icd_code'.
    """
    # d_icd_diagnoses = read_d_icd_diagnoses_table(path=path, compression=compression, nrows=nrows)

    if included_fname is not None and os.path.exists(os.path.join(cfg.path.included_feature_path, included_fname)):
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version']

    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, dataset=cfg.preprocess.dataset)
    # data = data.merge(d_icd_diagnoses, on='icd_code')

    return data


def read_procedures_icd_table(cfg,
                             path: str,
                             folder: str='hosp',
                             file_name: str='procedures_icd.csv.gz',
                             compression: str = 'gzip',
                             nrows: int = None,
                             included_fname: str='procedures_icd_included.txt') -> pd.DataFrame:
    """
    Read the procedures_icd table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from diagnoses_icd table and d_icd_diagnoses table. Two tables are merged on 'icd_code'.
    """

    if included_fname is not None and os.path.exists(os.path.join(cfg.path.included_feature_path, included_fname)):
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version']

    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, dataset=cfg.preprocess.dataset)

    return data


def read_transfers_table(cfg,
                         path: str,
                         folder: str='hosp',
                         file_name: str='transfers.csv.gz',
                         compression: str = 'gzip',
                         nrows: int = None,
                         included_fname: str='transfers_included.txt') -> pd.DataFrame:
    """
    Read the procedures_icd table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from diagnoses_icd table and d_icd_diagnoses table. Two tables are merged on 'icd_code'.
    """

    if included_fname is not None and os.path.exists(os.path.join(cfg.path.included_feature_path, included_fname)):
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = None

    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, dataset=cfg.preprocess.dataset)

    return data


def read_emar_table(cfg,
                         path: str,
                         folder: str='hosp',
                         file_name: str='emar.csv.gz',
                         compression: str = 'gzip',
                         nrows: int = None,
                         included_fname: str='emar_included.txt') -> pd.DataFrame:
    """
    Read the procedures_icd table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from diagnoses_icd table and d_icd_diagnoses table. Two tables are merged on 'icd_code'.
    """

    if included_fname is not None and os.path.exists(os.path.join(cfg.path.included_feature_path, included_fname)):
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = None

    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, dataset=cfg.preprocess.dataset)

    return data


def read_emar_detail_table(cfg,
                         path: str,
                         folder: str='hosp',
                         file_name: str='emar_detail.csv.gz',
                         compression: str = 'gzip',
                         nrows: int = None,
                         included_fname: str='emar_detail_included.txt') -> pd.DataFrame:
    """
    Read the procedures_icd table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from diagnoses_icd table and d_icd_diagnoses table. Two tables are merged on 'icd_code'.
    """

    if included_fname is not None and os.path.exists(os.path.join(cfg.path.included_feature_path, included_fname)):
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = None

    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, dataset=cfg.preprocess.dataset)

    return data

def read_labevents_table(cfg,
                         path: str,
                         folder: str='hosp',
                         file_name: str='labevents.csv.gz',
                         d_labitems_file_name: str='d_labitems.csv.gz',
                         compression: str = 'gzip',
                         nrows: int = None,
                         included_fname: str='labevents_included.txt') -> pd.DataFrame:
    """
    Read the labevents table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from labevents table and d_labitems table. Two tables are merged on 'itemid'.
    """

    d_labitems = read_d_labitems_table(path=path, folder=folder, file_name=d_labitems_file_name,
                                       compression=compression, nrows=nrows)

    if included_fname is not None and os.path.exists(os.path.join(cfg.path.included_feature_path, included_fname)):
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom']

    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, parse_dates=['charttime'], dataset=cfg.preprocess.dataset)
    data = data.merge(d_labitems, on='itemid')

    return data


def read_chartevents_table(cfg,
                           path: str,
                           folder: str='icu',
                           file_name: str='chartevents.csv.gz',
                           d_items_file_name: str='d_items.csv.gz',
                           compression: str = 'gzip',
                           nrows: int = None,
                           included_fname: str='chartevents_included.txt') -> pd.DataFrame:
    """
    Read the chartevents table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from chartevents table and d_items table. Two tables are merged on 'itemid'.
    """
    d_items = read_d_items_table(path=path, folder=folder, file_name=d_items_file_name,
                                 compression=compression, nrows=nrows, linksto='chartevents')

    if included_fname is not None and os.path.exists(os.path.join(cfg.path.included_feature_path, included_fname)):
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum', 'valueuom']

    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, parse_dates=['charttime'], dataset=cfg.preprocess.dataset)

    if 'item_id' in data.columns:
        data = data.rename(columns={'item_id': 'itemid'})
    data = data.merge(d_items, on='itemid')

    return data


def read_outputevents_table(cfg,
                            path: str,
                            folder: str='icu',
                            file_name: str='outputevents.csv.gz',
                            d_items_file_name: str='d_items.csv.gz',
                            compression: str = 'gzip',
                            nrows: int = None,
                            included_fname: str='outputevents_included.txt') -> pd.DataFrame:
    """
    Read the outputevents table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from outputevents table and d_items table. Two tables are merged on 'itemid'.
    """
    d_items = read_d_items_table(path=path, folder=folder, file_name=d_items_file_name,
                                 compression=compression, nrows=nrows, linksto='outputevents')

    if included_fname is not None and os.path.exists(os.path.join(cfg.path.included_feature_path, included_fname)):
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valueuom']
    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, parse_dates=['charttime'], dataset=cfg.preprocess.dataset)
    data = data.merge(d_items, on='itemid')

    return data


def read_procedureevents_table(cfg,
                            path: str,
                            folder: str='icu',
                            file_name: str='procedureevents.csv.gz',
                            d_items_file_name: str='d_items.csv.gz',
                            compression: str = 'gzip',
                            nrows: int = None,
                            included_fname: str='procedureevents_included.txt') -> pd.DataFrame:
    """
    Read the procedureevents table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from outputevents table and d_items table. Two tables are merged on 'itemid'.
    """
    d_items = read_d_items_table(path=path, folder=folder, file_name=d_items_file_name,
                                 compression=compression, nrows=nrows, linksto='procedureevents')

    if included_fname is not None and os.path.exists(os.path.join(cfg.path.included_feature_path, included_fname)):
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'itemid', 'patientweight']
    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, parse_dates=['starttime', 'endtime'], dataset=cfg.preprocess.dataset)
    data = data.merge(d_items, on='itemid')

    return data


def read_inputevents_table(cfg,
                            path: str,
                            folder: str='icu',
                            file_name: str='inputevents.csv.gz',
                            d_items_file_name: str='d_items.csv.gz',
                            compression: str = 'gzip',
                            nrows: int = None,
                            included_fname: str='inputevents_included.txt') -> pd.DataFrame:
    """
    Read the inputevents table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from inputevents table and d_items table. Two tables are merged on 'itemid'.
    """
    d_items = read_d_items_table(path=path, folder=folder, file_name=d_items_file_name,
                                 compression=compression, nrows=nrows, linksto='inputevents')

    if included_fname is not None and os.path.exists(os.path.join(cfg.path.included_feature_path, included_fname)):
        with open(os.path.join(cfg.path.included_feature_path, included_fname), 'r') as f:
            cols = [col.strip() for col in f.readlines()]
    else:
        cols = ['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'itemid', 'patientweight']
    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows, parse_dates=['starttime', 'endtime'], dataset=cfg.preprocess.dataset)
    data = data.merge(d_items, on='itemid')

    return data


def read_d_items_table(path: str,
                       folder: str='icu',
                       file_name: str='d_items.csv.gz',
                       compression: str='gzip', nrows: int=None, linksto: str or list=None, keep: bool=False) -> pd.DataFrame:
    """
    Read the d_items table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)
        linksto: table name to link to (default: None)

    Returns:
        pd.DataFrame: data from d_items table
    """
    cols = ['itemid', 'label', 'linksto']
    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows)
    if linksto:
        if isinstance(linksto, list):
            data = data[data['linksto'].isin(linksto)]
        elif isinstance(linksto, str):
            data = data[data['linksto'] == linksto]
        else:
            raise ValueError('linksto should be str or list')

    if not keep:
        data = data.drop(columns=['linksto'])

    data = data.drop_duplicates(keep='first').reset_index(drop=True)

    return data


def read_d_labitems_table(path: str,
                          folder: str='hosp',
                          file_name: str='d_labitems.csv.gz',
                          compression: str = 'gzip',
                          nrows: int = None) -> pd.DataFrame:
    """
    Read the d_labitems table from the MIMIC-IV dataset.
    Args:
        path: file path
        compression: compression type (default: 'gzip')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: data from d_labitems table
    """
    cols = ['itemid', 'label']
    path = os.path.join(path, folder, file_name)
    data = read_file(path, usecols=cols, compression=compression, nrows=nrows)

    data = data.drop_duplicates(keep='first').reset_index(drop=True)

    return data


# def read_icd9_to_icd10_mapping_table(path: str,
#                                      compression: str='infer',
#                                      nrows: int=None,
#                                      delimiter: str='\t') -> pd.DataFrame:
#     """
#     Read the ICD-9 to ICD-10 mapping table.
#     mapping table ref link: https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/utils/mappings/ICD9_to_ICD10_mapping.txt
#     Args:
#         path: file path of the mapping table
#         fname: file name of the mapping table (default: 'ICD9_to_ICD10_mapping.txt')
#         compression: compression type (default: 'infer')
#         nrows: number of rows to read (default: None)
#         delimiter: delimiter (default: '\t')
#
#     Returns:
#         pd.DataFrame: ICD-9 to ICD-10 mapping data
#     """
#     cols = ['diagnosis_code', 'icd9cm', 'icd10cm']
#     if nrows:
#         data = pd.read_csv(path, usecols=cols, compression=compression, nrows=nrows, delimiter=delimiter)
#     else:
#         data = pd.read_csv(path, usecols=cols, compression=compression, delimiter=delimiter)
#
#     return data

def read_icd9_to_icd10_mapping_table(path: str,
                                     compression: str='infer',
                                     nrows: int=None,
                                     delimiter: str=',') -> pd.DataFrame:
    """
    Read the ICD-9 to ICD-10 mapping table.
    mapping table ref link: https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/utils/mappings/ICD9_to_ICD10_mapping.txt
    Args:
        path: file path of the mapping table
        fname: file name of the mapping table (default: 'ICD9_to_ICD10_mapping.txt')
        compression: compression type (default: 'infer')
        nrows: number of rows to read (default: None)
        delimiter: delimiter (default: '\t')

    Returns:
        pd.DataFrame: ICD-9 to ICD-10 mapping data
    """
    cols = ['icd_9', 'icd_10']
    if nrows:
        data = pd.read_csv(path, usecols=cols, compression=compression, nrows=nrows, delimiter=delimiter)
    else:
        data = pd.read_csv(path, usecols=cols, compression=compression, delimiter=delimiter)

    data = data.drop_duplicates(keep='first').reset_index(drop=True)
    data['icd_9'] = data['icd_9'].astype(str)
    data['icd_10'] = data['icd_10'].astype(str)
    return data


def read_cohort(path: str, fname: str='cohort.csv', compression: str='infer', nrows: int=None, dataset: str=None) -> pd.DataFrame:
    """
    Read the cohort file
    Args:
        path: file path of the cohort file
        fanme: file name of the cohort file (default: 'cohort.csv')
        compression: compression type (default: 'infer')
        nrows: number of rows to read (default: None)

    Returns:
        pd.DataFrame: cohort data
    """
    path = os.path.join(path, fname)
    if nrows:
        data = pd.read_csv(path, compression=compression, nrows=nrows,
                           parse_dates=['admittime', 'dischtime', 'intime', 'outtime'], date_format='%Y-%m-%d %H:%M:%S.%f')
    else:
        data = pd.read_csv(path, compression=compression,
                           parse_dates=['admittime', 'dischtime', 'intime', 'outtime'], date_format='%Y-%m-%d %H:%M:%S.%f')

    data = convert_keys_type(data, dataset=dataset)

    return data


def read_features(path: str,
                  fname: str,
                  compression: str='infer',
                  nrows: int=None,
                  header: int=None,
                  delimiter: str='\t',
                  col_name: str='features') -> list:
    """
    Read the features file
    Args:
        path: file path of the features file
        fname: file name of the features file (default: 'continuous_valued_features.csv')
        compression: compression type (default: 'infer')
        nrows: number of rows to read (default: None)
        delimiter: delimiter (default: '\n')

    Returns:
        list: list of features data
    """
    path = os.path.join(path, fname)
    if nrows:
        data = pd.read_csv(path, compression=compression, nrows=nrows, header=header, delimiter=delimiter)
    else:
        data = pd.read_csv(path, compression=compression, header=header, delimiter=delimiter)

    data.columns = [col_name]
    data = data[col_name].values.tolist()

    return data


def read_item_id_stat(path: str,
                      compression: str='infer',
                      nrows: int=None,
                      id_col: str='itemid',
                      levels: list=['LEVEL1', 'LEVEL2']) -> pd.DataFrame:
    """
    Read the item_id_stat file
    itemid mapping table ref link: https://github.com/MLforHealth/MIMIC_Extract/blob/master/resources/item_id_stat.csv

    Args:
        path:
        fname:
        compression:
        nrows:
        id_col:
        levels:

    Returns:

    """
    path = os.path.join(path)
    if nrows:
        data = pd.read_csv(path, compression=compression, nrows=nrows)
    else:
        data = pd.read_csv(path, compression=compression)

    data = data[[id_col] + levels]

    return data


def load_data(path: str, fname: str, compression: str='infer', nrows: int=None) -> pd.DataFrame:
    """
    Read the diagnoses file
    Args:
        path:
        fname:
        compression:
        nrows:

    Returns:

    """
    path = os.path.join(path, fname)
    if nrows:
        data = pd.read_csv(path, compression=compression, nrows=nrows)
    else:
        data = pd.read_csv(path, compression=compression)

    return data


def read_events(path: str, fname: str='events.csv', compression: str='infer', nrows: int=None) -> pd.DataFrame:
    """
    Read the events file
    Args:
        path:
        fname:
        compression:
        nrows:

    Returns:

    """
    path = os.path.join(path, fname)
    if nrows:
        data = pd.read_csv(path, compression=compression, nrows=nrows)
    else:
        data = pd.read_csv(path, compression=compression)

    return data


def read_static_data(path: str, fname: str='static_data.csv', compression: str='infer', nrows: int=None) -> pd.DataFrame:
    """
    Read the static data file
    Args:
        path:
        fname:
        compression:
        nrows:

    Returns:

    """

    path = os.path.join(path, fname)
    if nrows:
        data = pd.read_csv(path, compression=compression, nrows=nrows)
    else:
        data = pd.read_csv(path, compression=compression)

    return data


def save_csv(data: pd.DataFrame, path: str, fname: str, index: bool=False) -> None:
    """
    Save the data to the specified path.
    Args:
        data: data to save
        path: path to save the data
        fname: file name to save the data
        index: whether to save the index (default: False)

    Returns:

    """

    os.makedirs(path, exist_ok=True)
    data.to_csv(os.path.join(path, fname), index=index)

