from datetime import datetime
import re

import pandas as pd
from tqdm import tqdm
import numpy as np

from Preprocess.KMIMIC.utils.utils import *
from Preprocess.KMIMIC.utils.file import *
import config_manager


@measure_runtime
def extract_diagnoses_icd(cfg):
    """
    Extract diagnoses from the original data
    Args:
        cfg: configuration

    Returns:

    """
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)

    # load data
    cohort = read_cohort(preprocessed_data_path, nrows=debug_num)

    # get diagnoses on cohort
    if cfg.preprocess.flag.diagnoses_icd:
        diagnoses = get_diagnoses_icd_on_cohort(cfg, data_path, cohort,
                                            file_name='diagnoses_icd.parquet',
                                            verbose=cfg.preprocess.verbose)
        # diagnoses = preprocess_icd_code(diagnoses, path=cfg.path.mapping.icd9_to_icd10_map, verbose=cfg.preprocess.verbose)
        diagnoses = convert_col_on_frequency_rank(diagnoses, col_name='icd_code', threshold=cfg.preprocess.icd_code.ratio)
        diagnoses = pivot_icd(diagnoses, key_cols=cfg.preprocess.keys, col_name='icd_code',
                              icd_code_col_pattern=cfg.preprocess.icd_code.diagnoses_prefix)

        # check duplicated value
        diagnoses = check_duplicated_value(diagnoses)
        # save diagnoses file
        save_csv(diagnoses, preprocessed_data_path, 'diagnoses.csv')


@measure_runtime
def extract_procedures_icd(cfg):
    """
    Extract diagnoses from the original data
    Args:
        cfg: configuration

    Returns:

    """
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)

    # load data
    cohort = read_cohort(preprocessed_data_path, nrows=debug_num)

    # get procedures on cohort
    if cfg.preprocess.flag.procedures_icd:
        procedures = get_procedures_icd_on_cohort(cfg, data_path, cohort,
                                            file_name='procedures_icd.parquet',
                                            verbose=cfg.preprocess.verbose)
        procedures = convert_col_on_frequency_rank(procedures, threshold=cfg.preprocess.icd_code.ratio)
        procedures = pivot_icd(procedures, key_cols=cfg.preprocess.keys,
                               icd_code_col_pattern=cfg.preprocess.icd_code.procedure_prefix)

        # check duplicated value
        procedures = check_duplicated_value(procedures)
        # save diagnoses file
        save_csv(procedures, preprocessed_data_path, 'procedures.csv')


def extract_transfers(cfg):
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)

    # load data
    cohort = read_cohort(preprocessed_data_path, nrows=debug_num)

    # get transfers on cohort
    if cfg.preprocess.flag.transfers:
        data = get_transfers_on_cohort(cfg, data_path, cohort,
                                             file_name='transfers.parquet',
                                             verbose=cfg.preprocess.verbose)
        data = convert_col_on_frequency_rank(data, col_name='careunit', threshold=cfg.preprocess.cutoff_frequency_rank_ratio)
        data = one_hot_encode(data, col_name='careunit', prefix='careunit')

        # binary encode the transfer status
        data['transfer'] = data['eventtype'].apply(lambda x: 1 if x == 'transfer' else 0)

        # check duplicated value
        data = check_duplicated_value(data)
        # save diagnoses file
        save_csv(data, preprocessed_data_path, 'transfers.csv')


def unit_conversion(row):
    value = row['infusion_rate']
    weight = row['patientweight']
    unit = str(row['infusion_rate_unit']).lower().strip() if pd.notna(row['infusion_rate_unit']) else None
    standard_unit = str(row['standard_unit']).lower().strip() if pd.notna(row['standard_unit']) else None

    if pd.isna(value) or pd.isna(unit) or pd.isna(standard_unit):
        return None

    medication_rules = {
        'Norepinephrine': lambda v, w, u, su: convert_norepinephrine(v, w, u, su),
        'Remifentanil': lambda v, w, u, su: convert_remifentanil(v, w, u, su),
        'Propofol': lambda v, w, u, su: convert_propofol(v, w, u, su),
    }

    medication = row['medication']
    if medication in medication_rules:
        return medication_rules[medication](value, weight, unit, standard_unit)

    return general_unit_conversion(value, weight, unit, standard_unit)


def convert_norepinephrine(value, weight, unit, standard_unit):
    if standard_unit == 'mcg/min':
        if unit == 'mcg/min':
            return value
        elif unit == 'mcg/kg/min':
            if value < 1:
                return value * weight if weight else None
            else:
                return value
        else:
            match = re.match(r'mcg/([\d.]+)kg/min', unit)
            if match:
                w = float(match.group(1))
                return value * w
    return None


def convert_remifentanil(value, weight, unit, standard_unit):
    if standard_unit == 'mcg/kg/min':
        if unit == 'mcg/kg/min':
            return 0.05 if value >= 5 else value
        elif unit == 'mcg/min':
            if value < 1:
                return value
            else:
                return value / weight if weight else None
        else:
            match = re.match(r'mcg/([\d.]+)kg/min', unit)
            if match:
                w = float(match.group(1))
                return value
    return None


def convert_propofol(value, weight, unit, standard_unit):
    if standard_unit == 'mcg/kg/min':
        if unit == 'mcg/kg/min' or unit == 'mcg/min':
            return value
        else:
            match = re.match(r'mcg/([\d.]+)kg/min', unit)
            if match:
                w = float(match.group(1))
                return value
    return None


def general_unit_conversion(value, weight, unit, standard_unit):
    """
    General conversion logic for medications without specific rules.
    """
    cc_like_units = ['cc/hr', 'c/hr', 'ccc/hr', 'cch/hr', 'cxc/hr']

    if standard_unit == 'mcg/kg/min':
        if unit == 'mcg/kg/min':
            return value
        elif unit in cc_like_units:
            return None
        elif unit == 'mcg/min':
            return value / weight if weight else None
        elif unit == 'mcg/hr':
            return (value / 60) / weight if weight else None
        elif unit == 'mg/hr':
            return (value * 1000 / 60) / weight if weight else None
        else:
            match = re.match(r'mcg/([\d.]+)kg/min', unit)
            if match:
                return value
    elif standard_unit == 'mg/hr':
        if unit == 'mg/hr':
            return value
        elif unit in cc_like_units:
            return None
        elif unit == 'mcg/min':
            return value * 60 / 1000
        elif unit == 'mcg/kg/min':
            return value * weight * 60 / 1000 if weight else None
        elif unit == 'mcg/hr':
            return value / 1000

    elif standard_unit == 'cc/hr':
        if unit in cc_like_units:
            return value

    elif standard_unit == 'unit/hr':
        if unit == 'unit/hr':
            return value
        elif unit == 'unit/min':
            return value * 60

    elif standard_unit == 'unit/min':
        if unit == 'unit/min':
            return value
        elif unit == 'unit/hr':
            return value / 60

    return None


def unit_conversion_propofol(row):
    value = row['infusion_rate']
    weight = row['patientweight']
    unit = row['infusion_rate_unit']
    standard_unit = row['standard_unit']

    if pd.isna(value) or pd.isna(unit) or pd.isna(standard_unit):
        return None

    unit = str(unit).lower().strip()
    standard_unit = str(standard_unit).lower().strip()
    value = float(value)

    if standard_unit == 'mcg/kg/min':
        if unit == 'mcg/kg/min':
            return value
        elif unit == 'mcg/min':
            return value
        else:
            match = re.match(r'mcg/([\d.]+)kg/min', unit)
            if match:
                w = float(match.group(1))
                return value
            else:
                return None


def clean_units(unit):
    if pd.isna(unit):
        return None
    unit = unit.lower().strip()
    unit_mappings = {
        'c/hr': 'cc/hr',
        'ccc/hr': 'cc/hr',
        'cxc/hr': 'cc/hr',
        'units/hr': 'unit/hr',
        'u/hr': 'unit/hr',
        'unti/hr': 'unit/hr',
        'ui/hr': 'unit/hr',
    }

    for error_unit, standard_unit in unit_mappings.items():
        if unit in error_unit:
            return standard_unit
    return unit


def is_valid_unit(x):
    if x is None:
        return False
    if isinstance(x, float):
        return not np.isnan(x)
    return True


def merge_infusion_rate_or_unit(row, return_type='value'):
    """
    Merge infusion rate or unit based on the standard unit.

    Args:
        row: A row of the DataFrame.
        return_type: Specify whether to return 'value' or 'unit'.

    Returns:
        The merged infusion rate value or unit, or None if invalid.
    """
    if is_valid_unit(row['standard_unit']):
        if 'hr' in row['standard_unit']:
            return row[f'infusion_rate_hr_{return_type}']
        elif 'min' in row['standard_unit']:
            return row[f'infusion_rate_min_{return_type}']
    return None


def extract_infusion_rate_value_and_unit(x, time_unit):
    """
    Extract value and unit from infusion rate string based on the time unit (e.g., '/min' or '/hr').
    Args:
        x: Input string containing infusion rate.
        time_unit: Time unit to match ('/min' or '/hr').

    Returns:
        Tuple of (value, unit) if matched, otherwise (None, None).
    """
    pattern1 = rf'^(\d+(?:\.\d+)?)\s*([a-zA-Z]+(?:/\d*\.?\d*kg|/kg)?{time_unit})$'
    pattern2 = rf'(\d+(?:\.\d+)?)\s*([a-zA-Z]+(?:/\d*\.?\d*kg|/kg)?{time_unit})'

    x = str(x)
    match = re.search(pattern1, x) or re.search(pattern2, x)
    if match:
        return match.group(1), match.group(2)
    return None, None


def filter_infusion_rate(data, bounds):
    """
    Filter the data based on medication-specific infusion rate bounds.

    Args:
        data: DataFrame containing medication and infusion rate columns.
        bounds: Dictionary with medication as keys and (low, high) bounds as values.

    Returns:
        Filtered DataFrame.
    """
    for med, (low, high) in bounds.items():
        if low is not None:
            data = data[~((data['medication'] == med) & (data['infusion_rate'] < low))]
        if high is not None:
            data = data[~((data['medication'] == med) & (data['infusion_rate'] > high))]
    return data

def process_infusion_rate(data):
    """
    Process infusion rates for specific medications.

    Args:
        data: DataFrame containing medication and infusion rate columns.

    Returns:
        DataFrame with processed infusion rates.
    """
    data['infusion_rate'] = data.apply(
        lambda x: process_dexmedetomidine(x['infusion_rate']) if x['medication'] == 'Dexmedetomidine' else x['infusion_rate'],
        axis=1
    )
    return data


def process_dexmedetomidine(value):
    if value >= 20:
        return value / 100
    elif value >= 2:
        return value / 10
    else:
        return value


def extract_emar(cfg):
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)
    # load data
    cohort = read_cohort(preprocessed_data_path, nrows=debug_num)

    # get emar on cohort
    if cfg.preprocess.flag.emar:
        data = get_emar_on_cohort(cfg, data_path, cohort,
                                  file_name='emar.parquet',
                                  verbose=cfg.preprocess.verbose)
        emar_detail = read_emar_detail_table(cfg=cfg,
                                             path=data_path,
                                             folder='hosp', file_name='emar_detail.parquet')
        # only use 'Note' type of emar
        data = data[data['emar_type'] == 'Note']
        data['charttime'] = data['charttime'].map(parse_datetime)

        data = pd.merge(data,
                        emar_detail,
                        on=['subject_id', 'hadm_id', 'stay_id', 'emar_id', 'itemid', 'emar_seq', 'pharmacy_id'],
                        how='left')
        data = data[['subject_id', 'hadm_id', 'stay_id', 'medication', 'charttime',
                     'event_txt', 'note_txt', 'infusion_rate_hr', 'infusion_rate_min']]

        # mapping table
        mapping_path = os.path.join(cfg.path.mapping_path, 'emar_count.xlsx')
        label_map = load_label_mapping_table(mapping_path)
        data['medication'] = data['medication'].map(lambda x: label_map[x] if x in label_map.keys() else None)
        data = data[data['medication'].notnull()]

        medication_standard_unit_map = pd.read_excel(os.path.join(cfg.path.mapping_path, 'emar_standard_unit.xlsx'))
        medication_standard_unit_map = medication_standard_unit_map.rename(columns={'label_modified': 'medication',
                                                                                    'unit': 'standard_unit'})
        data = data[data['medication'] != 'Others']
        data = pd.merge(data, medication_standard_unit_map, on='medication', how='left')

        data['infusion_rate_min_value'], data['infusion_rate_min_unit'] = zip(
            *data['note_txt'].map(lambda x: extract_infusion_rate_value_and_unit(x, '/min'))
        )
        data['infusion_rate_hr_value'], data['infusion_rate_hr_unit'] = zip(
            *data['note_txt'].map(lambda x: extract_infusion_rate_value_and_unit(x, '/hr'))
        )

        # data['infusion_rate_min_value'], data['infusion_rate_min_unit'] = zip(
        #     *data['infusion_rate_min'].map(lambda x: extract_infusion_rate_value_and_unit(x, '/min'))
        # )
        # data['infusion_rate_hr_value'], data['infusion_rate_hr_unit'] = zip(
        #     *data['infusion_rate_hr'].map(lambda x: extract_infusion_rate_value_and_unit(x, '/hr'))
        # )
        data['infusion_rate_hr_unit'] = data['infusion_rate_hr_unit'].map(clean_units)
        data['infusion_rate_min_unit'] = data['infusion_rate_min_unit'].map(clean_units)

        data['infusion_rate'] = data.apply(lambda row: merge_infusion_rate_or_unit(row, return_type='value'), axis=1)
        data['infusion_rate_unit'] = data.apply(lambda row: merge_infusion_rate_or_unit(row, return_type='unit'), axis=1)
        data = data[data['infusion_rate'].notnull() & data['infusion_rate_unit'].notnull()]

        # merge patient weight
        procedureevents_weights = pd.read_csv(os.path.join(preprocessed_data_path, 'procedureevents.csv'), usecols=['subject_id', 'patientweight'])
        inputevents_weights = pd.read_csv(os.path.join(preprocessed_data_path, 'inputevents.csv'), usecols=['subject_id', 'patientweight'])
        weights = pd.concat([procedureevents_weights, inputevents_weights], axis=0)
        weights = weights.groupby('subject_id')['patientweight'].mean()
        data = data.merge(weights, on=['subject_id'], how='left')

        data['infusion_rate'] = data['infusion_rate'].astype(float)
        data['infusion_rate'] = data.apply(unit_conversion, axis=1)
        data = data[data['infusion_rate'].notnull()]
        data = data[data['infusion_rate'] > 0]

        bounds = {'Plasma solution A': (0.11, None), 'Remifentanil': (None, 2)}
        data = filter_infusion_rate(data, bounds)
        data = process_infusion_rate(data)

        # check duplicated value
        data = check_duplicated_value(data)
        data = data[['subject_id', 'hadm_id', 'stay_id', 'medication', 'charttime', 'event_txt', 'infusion_rate',
                     'infusion_rate_unit', 'standard_unit']]

        # save diagnoses file
        save_csv(data, preprocessed_data_path, 'emar.csv')
        # data = data.groupby(['subject_id','hadm_id','stay_id','medication'])['charttime'].agg(starttime='min', endtime='max').reset_index()


def preprocess_gcs(data):
    def safe_int(x):
        try:
            return int(x)
        except:
            return None

    gcs_score_dict = {
        'Glasgow coma scale(eye)': list(range(1, 5)),
        'Glasgow coma scale(verbal)': list(range(1, 6)),
        'Glasgow coma scale(motor)': list(range(1, 7))
    }

    for k, v in gcs_score_dict.items():
        data.loc[data['label'] == k, 'value'] = data.loc[data['label'] == k, 'value'].apply(
            lambda x: x if safe_int(x) in v else None
        )

    return data


def extract_chartevents(cfg):
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)
    chunk_size = 500000
    verbose = cfg.preprocess.verbose

    # load data
    cohort = read_cohort(preprocessed_data_path, nrows=debug_num)

    chartevents = get_chartevents_on_cohort(cfg, data_path, cohort,
                                            folder='icu',
                                            file_name='chartevents.csv',
                                            d_items_file_name='d_items.csv',
                                            compression=None,
                                            # file_name='chartevents.parquet',
                                            # d_items_file_name='d_items.parquet',
                                            nrows=debug_num,
                                            chunk_size=chunk_size,
                                            verbose=verbose)
    # chartevents = filter_event_items_by_ratio(chartevents)
    # mapping table
    mapping_path = os.path.join(cfg.path.mapping_path, 'chartevents_count.xlsx')
    label_map = load_label_mapping_table(mapping_path)
    chartevents = chartevents[chartevents['label'].isin(label_map.keys())]
    chartevents['label'] = chartevents['label'].map(lambda x: label_map[x] if x in label_map.keys() else None)
    chartevents = chartevents[chartevents['label'].notnull()]
    chartevents = chartevents[chartevents['label'] != 'Others']

    chartevents = preprocess_gcs(chartevents)

    not_vent_mode = chartevents[chartevents['label'] != 'Vent mode'].copy()
    not_vent_mode = not_vent_mode.drop(columns=['value'])
    not_vent_mode['value'] = not_vent_mode['valuenum']

    vent_mode = chartevents[chartevents['label'] == 'Vent mode'].copy()
    vent_mode = vent_mode.drop(columns=['valuenum'])

    chartevents = pd.concat([not_vent_mode, vent_mode], axis=0)
    # drop row with '-'
    chartevents = chartevents[~chartevents['value'].isin(['-'])]
    # drop row with None
    chartevents = chartevents[chartevents['value'].notnull()]

    if cfg.preprocess.cohort.only_icu_stays:
        chartevents = chartevents[chartevents['stay_id'].isin(cohort['stay_id'].unique())]

    chartevents = chartevents[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'value', 'label']]
    save_csv(chartevents, preprocessed_data_path, 'chartevents.csv')

    return chartevents


def extract_outputevents(cfg):
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)
    chunk_size = None
    verbose = cfg.preprocess.verbose

    # load data
    cohort = read_cohort(preprocessed_data_path, nrows=debug_num)

    outputevents = get_outputevents_on_cohort(cfg, data_path, cohort,
                                              folder='icu',
                                              file_name='outputevents.parquet',
                                              d_items_file_name='d_items.parquet',
                                              nrows=debug_num,
                                              chunk_size=chunk_size, verbose=verbose)
    # outputevents = filter_event_items_by_ratio(outputevents)
    # mapping table
    mapping_path = os.path.join(cfg.path.mapping_path, 'outputevents_count.xlsx')
    label_map = load_label_mapping_table(mapping_path)
    outputevents = outputevents[outputevents['label'].isin(label_map.keys())]
    outputevents['label'] = outputevents['label'].map(lambda x: label_map[x] if x in label_map.keys() else None)
    outputevents = outputevents[outputevents['label'] != 'Others']
    outputevents = outputevents[outputevents['label'].notnull()]

    # drop row with '-'
    outputevents = outputevents[~outputevents['value'].isin(['-'])]
    # drop row with None
    outputevents = outputevents[outputevents['value'].notnull()]

    if cfg.preprocess.cohort.only_icu_stays:
        outputevents = outputevents[outputevents['stay_id'].isin(cohort['stay_id'].unique())]

    outputevents = outputevents[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'value', 'label']]
    outputevents = outputevents[outputevents['value'].map(is_number)]
    outputevents['value'] = outputevents['value'].astype(float)

    save_csv(outputevents, preprocessed_data_path, 'outputevents.csv')

    return outputevents


def extract_labevents(cfg):
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)
    chunk_size = None
    verbose = cfg.preprocess.verbose

    # load data
    cohort = read_cohort(preprocessed_data_path, nrows=debug_num)

    labevents = get_labevents_on_cohort(cfg, data_path, cohort,
                                        folder='hosp',
                                        file_name='labevents.parquet',
                                        d_labitems_file_name='d_labitems.parquet',
                                        nrows=debug_num,
                                        chunk_size=chunk_size, verbose=verbose)
    labevents = labevents[labevents['value'].notnull() | labevents['valuenum'].notnull()]
    # labevents = filter_event_items_by_ratio(labevents)
    # mapping table
    mapping_path = os.path.join(cfg.path.mapping_path, 'labevents_count.xlsx')
    label_map = load_label_mapping_table(mapping_path)
    labevents = labevents[labevents['label'].isin(label_map.keys())]
    labevents['label'] = labevents['label'].map(lambda x: label_map[x] if x in label_map.keys() else None)
    labevents = labevents[labevents['label'] != 'Others']
    labevents = labevents[labevents['label'].notnull()]
    labevents = labevents.drop(columns=['value'])
    labevents = labevents.rename(columns={'valuenum': 'value'})

    # drop row with '-'
    labevents = labevents[~labevents['value'].isin(['-'])]
    # drop row with None
    labevents = labevents[labevents['value'].notnull()]

    if cfg.preprocess.cohort.only_icu_stays:
        labevents = labevents[labevents['stay_id'].isin(cohort['stay_id'].unique())]

    labevents = labevents[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'value', 'label']]
    labevents = labevents[labevents['value'].map(is_number)]
    labevents['value'] = labevents['value'].astype(float)

    save_csv(labevents, preprocessed_data_path, 'labevents.csv')

    return labevents


def extract_inputevents(cfg):
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)
    chunk_size = None
    verbose = cfg.preprocess.verbose

    # load data
    cohort = read_cohort(preprocessed_data_path, nrows=debug_num)

    inputevents = get_inputevents_on_cohort(cfg, data_path, cohort,
                                                    folder='icu',
                                                    file_name='inputevents.parquet',
                                                    d_items_file_name='d_items.parquet',
                                                    nrows=debug_num,
                                                    chunk_size=chunk_size, verbose=verbose)
    # mapping table
    mapping_path = os.path.join(cfg.path.mapping_path, 'inputevents_count.xlsx')
    label_map = load_label_mapping_table(mapping_path)
    inputevents = inputevents[inputevents['label'].isin(label_map.keys())]
    inputevents['label'] = inputevents['label'].map(lambda x: label_map[x] if x in label_map.keys() else None)
    inputevents = inputevents[inputevents['label'] != 'Others']
    inputevents = inputevents[inputevents['label'].notnull()]

    row_with_no_num = inputevents[~inputevents['amount'].apply(is_number)]
    pattern = r'(\d+)\s*cc'
    row_with_no_num['amount'] = row_with_no_num['amount'].map(lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else None)
    row_with_num = inputevents[inputevents['amount'].apply(is_number)]
    inputevents = pd.concat([row_with_no_num, row_with_num], axis=0)
    inputevents['amount'] = inputevents['amount'].astype(float)

    # remove rows with string values in 'amount' column
    # inputevents = inputevents[inputevents['amount'].apply(is_number)]
    inputevents = inputevents[inputevents['amount'].notnull()]
    inputevents = inputevents.reset_index(drop=True)
    # inputevents = filter_event_items_by_ratio(inputevents)

    if cfg.preprocess.cohort.only_icu_stays:
        inputevents = inputevents[inputevents['stay_id'].isin(cohort['stay_id'].unique())]

    save_csv(inputevents, preprocessed_data_path, 'inputevents.csv')

    return inputevents


def extract_procedureevents(cfg):
    dataset = cfg.preprocess.dataset
    if cfg.preprocess.debug:
        debug_num = cfg.preprocess.debug_num
    else:
        debug_num = None

    data_path = os.path.join(cfg.path.raw_data_path, dataset)
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)
    chunk_size = None
    verbose = cfg.preprocess.verbose

    # load data
    cohort = read_cohort(preprocessed_data_path, nrows=debug_num)

    procedureevents = get_procedureevents_on_cohort(cfg, data_path, cohort,
                                                    folder='icu',
                                                    file_name='procedureevents.parquet',
                                                    d_items_file_name='d_items.parquet',
                                                    nrows=debug_num,
                                                    chunk_size=chunk_size, verbose=verbose)
    # mapping table
    mapping_path = os.path.join(cfg.path.mapping_path, 'procedureevents_count.xlsx')
    label_map = load_label_mapping_table(mapping_path)
    procedureevents = procedureevents[procedureevents['label'].isin(label_map.keys())]
    procedureevents['label'] = procedureevents['label'].map(lambda x: label_map[x] if x in label_map.keys() else None)
    procedureevents = procedureevents[procedureevents['label'] != 'Others']
    procedureevents = procedureevents[procedureevents['label'].notnull()]

    if cfg.preprocess.cohort.only_icu_stays:
        procedureevents = procedureevents[procedureevents['stay_id'].isin(cohort['stay_id'].unique())]

    save_csv(procedureevents, preprocessed_data_path, 'procedureevents.csv')

    return procedureevents


def extract_events(cfg):
    dataset = cfg.preprocess.dataset
    verbose = cfg.preprocess.verbose
    preprocessed_data_path = os.path.join(cfg.path.preprocessed_data_path, dataset)

    events = []
    if cfg.preprocess.flag.chartevents:
        print("Extracting chartevents...")
        chartevents = extract_chartevents(cfg=cfg)
        events.append(chartevents)

    if cfg.preprocess.flag.outputevents:
        print("Extracting outputevents...")
        outputevents = extract_outputevents(cfg=cfg)
        events.append(outputevents)

    if cfg.preprocess.flag.labevents:
        print("Extracting labevents...")
        labevents = extract_labevents(cfg=cfg)
        events.append(labevents)

    if len(events) > 0:
        events = pd.concat(events, axis=0)

        print('COMPLETE READ EVENT FILE:\n\tsubject_id: {}\n\thadm_id: {}\n\tstay_id: {}'.format(
            events.subject_id.unique().shape[0],
            events.hadm_id.unique().shape[0],
            events.stay_id.unique().shape[0])
        )

        events = check_duplicated_value(events, drop=True)

        # drop nan value
        events = events.dropna(subset=['value'])

        # save events file
        save_csv(events, preprocessed_data_path, 'events.csv')
    else:
        assert False, 'No events data'

    if verbose:
        print('COMPLETE SAVE EVENTS FILE:\n\tsubject_id: {}\n\thadm_id: {}\n\tstay_id: {}'.format(
            events.subject_id.unique().shape[0],
            events.hadm_id.unique().shape[0],
            events.stay_id.unique().shape[0])
        )


if __name__ == "__main__":
    config_manager.load_config()
    cfg = config_manager.config

    if cfg.preprocess.flag.diagnoses_icd:
        print("Extracting diagnoses ICD...")
        extract_diagnoses_icd(cfg=cfg)

    if cfg.preprocess.flag.procedures_icd:
        print("Extracting procedures ICD...")
        extract_procedures_icd(cfg=cfg)

    if cfg.preprocess.flag.transfers:
        print("Extracting transfers...")
        extract_transfers(cfg=cfg)

    if cfg.preprocess.flag.chartevents:
        print("Extracting chartevents...")
        chartevents = extract_chartevents(cfg=cfg)

    if cfg.preprocess.flag.outputevents:
        print("Extracting outputevents...")
        outputevents = extract_outputevents(cfg=cfg)

    if cfg.preprocess.flag.labevents:
        print("Extracting labevents...")
        labevents = extract_labevents(cfg=cfg)

    if cfg.preprocess.flag.procedureevents:
        print("Extracting procedureevents...")
        extract_procedureevents(cfg=cfg)

    if cfg.preprocess.flag.inputevents:
        print("Extracting inputevents...")
        extract_inputevents(cfg=cfg)

    if cfg.preprocess.flag.emar:
        print("Extracting emar...")
        extract_emar(cfg=cfg)