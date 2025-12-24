import os
import pandas as pd
import numpy as np
from typing import List
import h5py
from tqdm import tqdm
from omegaconf import DictConfig
from pandarallel import pandarallel

from Utils.utils import measure_runtime


def csv_to_parquet(fpath: str, save_fpath: str):
    data = pd.read_csv(fpath)
    data.to_parquet(save_fpath, engine='pyarrow')


def check_column_type(data: pd.DataFrame,
                      dropna: bool = True,
                      threshold: int = 7):
    feature_types = {}

    for col in data.columns:
        col_data = data[col]
        col_data_no_na = col_data.dropna() if dropna else col_data

        if pd.api.types.is_bool_dtype(col_data):
            feature_types[col] = 'Binary'

        elif pd.api.types.is_numeric_dtype(col_data):
            unique_vals = col_data_no_na.unique()

            if set(unique_vals).issubset({0, 1}):
                feature_types[col] = 'Binary'
            elif (np.all(col_data_no_na % 1 == 0) and
                  col_data_no_na.nunique() <= threshold):
                feature_types[col] = 'Categorical'
            else:
                feature_types[col] = 'Numerical'

        elif pd.api.types.is_object_dtype(col_data):
            feature_types[col] = 'Categorical'
        else:
            feature_types[col] = 'Other'

    return feature_types



def get_column_dtype(data: pd.DataFrame):
    dtypes = data.dtypes.to_dict()

    return dtypes


def convert_to_original_dtype(data: pd.DataFrame, dtypes: dict, round_decimal: int=2):
    for col, _dtype in dtypes.items():
        try:
            data[col] = data[col].astype(_dtype)

        except KeyError:
            pass
        except Exception as e:
            print(col, _dtype, data[col].dtype)
            pass

    float_cols = data.select_dtypes(include=['float32', 'float64']).columns
    data[float_cols] = data[float_cols].round(round_decimal)

    return data


def impute_missing_values(data: pd.DataFrame,
                          key_cols: List[str],
                          imputation_agg_func: str,
                          feature_type: dict):
    numerical_cols = [k for k, v in feature_type.items() if v == 'Numerical']
    categorical_cols = [k for k, v in feature_type.items() if v == 'Categorical']
    binary_cols = [k for k, v in feature_type.items() if v == 'Binary']

    result_data = data.copy()
    if len(numerical_cols) > 0:
        numerical_data = result_data[numerical_cols]
        group_agg_value = numerical_data.groupby(level=key_cols, group_keys=False, dropna=False).agg(imputation_agg_func)
        feature_agg_value = numerical_data.agg(imputation_agg_func)
        # Imputation with interpolation, ffill, and group-level fill
        numerical_data = numerical_data.groupby(level=key_cols, group_keys=False, dropna=False).apply(
            lambda group: group.interpolate(method='linear').ffill()
        )
        numerical_data = numerical_data.fillna(group_agg_value).fillna(feature_agg_value)
        result_data[numerical_cols] = numerical_data

    if len(categorical_cols) > 0:
        result_data[categorical_cols] = result_data[categorical_cols].fillna('NA')

    if len(binary_cols) > 0:
        result_data[binary_cols] = result_data[binary_cols].fillna(0)

    return result_data


@measure_runtime
def preprocess_operations(cfg,
                          data_path: str,
                          fname: str='operations.csv.parquet',
                          excluded_fname: str='operations_excluded.txt',
                          imputation: bool = True,
                          outliers: bool=True,
                          outlier_threshold: dict=None,
                          imputation_agg_func: str = 'mean'):
    if os.path.splitext(fname)[-1] == '.csv':
        data = pd.read_csv(os.path.join(data_path, fname))
    elif os.path.splitext(fname)[-1] == '.parquet':
        data = pd.read_parquet(os.path.join(data_path, fname))
    else:
        raise ValueError('Unknown file format')

    original_dtypes = get_column_dtype(data)

    key_cols = [col for col in data.columns if col in cfg.preprocess.key_cols]

    data['los'] = data['discharge_time'] - data['admission_time']
    data['icu_los'] = data['icuout_time'] - data['icuin_time']

    # 1: death, 0: alive
    data['inhosp_mortality'] = data['inhosp_death_time'].notnull()
    data['allcause_mortality'] = data['allcause_death_time'].notnull()

    if excluded_fname is not None:
        with open(os.path.join(cfg.path.excluded_feature_path, excluded_fname), 'r') as f:
            excluded_features = [col.strip() for col in f.readlines()]
        data = data.drop(columns=excluded_features)


    mask = data.set_index(key_cols).map(lambda x: 0 if pd.isnull(x) else 1).reset_index()

    if outliers:
        data, mask = remove_outliers(data, mask, outlier_threshold)

    if imputation:
        data = impute_missing_values(data=data.set_index(key_cols),
                                     key_cols=key_cols,
                                     imputation_agg_func=imputation_agg_func,
                                     feature_type=feature_type)
    data = data.reset_index()

    return data, mask, feature_type, original_dtypes


@measure_runtime
def preprocess_observations(cfg,
                            data_path: str,
                            fname: str='vitals.csv.parquet',
                            excluded_fname: str=None,
                            imputation: bool=True,
                            imputation_agg_func: str='mean',
                            outliers: bool=True,
                            outlier_threshold: dict=None,
                            row_threshold_ratio: float=0.01,
                            sample_patients: pd.DataFrame=None):
    if os.path.splitext(fname)[-1] == '.csv':
        data = pd.read_csv(os.path.join(data_path, fname))
    elif os.path.splitext(fname)[-1] == '.parquet':
        data = pd.read_parquet(os.path.join(data_path, fname))
    else:
        raise ValueError('Unknown file format')

    original_dtypes = get_column_dtype(data)

    key_cols = [col for col in data.columns if col in cfg.preprocess.key_cols]
    time_cols = cfg.preprocess.time_cols

    if cfg.dataset.debug:
        if sample_patients is not None:
            data = data.merge(sample_patients[key_cols], on=key_cols)
        else:
            data = data.sample(cfg.dataset.debug_n)

    if excluded_fname is not None:
        with open(os.path.join(cfg.path.excluded_feature_path, excluded_fname), 'r') as f:
            excluded_features = [col.strip() for col in f.readlines()]
        data = data.drop(columns=excluded_features)

    # drop columns that have not enough rows
    if row_threshold_ratio is not None:
        row_threshold = data.shape[0] * row_threshold_ratio
        unique_item_count = data['item_name'].value_counts()
        keep_item = unique_item_count[unique_item_count >= row_threshold]
        data = data[data['item_name'].isin(keep_item.index)]

    data = pd.pivot_table(data, index=key_cols+time_cols, columns='item_name', values='value')
    feature_type = check_column_type(data)
    mask = data.map(lambda x: 0 if pd.isnull(x) else 1)

    if outliers:
        data, mask = remove_outliers(data, mask, outlier_threshold)

    if imputation:
        data = impute_missing_values(data=data,
                                     key_cols=key_cols,
                                     imputation_agg_func=imputation_agg_func,
                                     feature_type=feature_type)

    data = data.reset_index()
    mask = mask.reset_index()

    return data, mask, feature_type, key_cols, original_dtypes


@measure_runtime
def preprocess_medications(cfg,
                           data_path: str,
                           fname: str='medications.csv.parquet',
                           excluded_fname: str='medications_excluded.txt',
                           drug_name_cols: List[str]=['drug_name', 'drug_name2', 'drug_name3'],
                           value_col: str='route',
                           imputation: bool=True,
                           imputation_agg_func: str='mean',
                           outliers: bool=True,
                           outlier_threshold: dict=None,
                           row_threshold_ratio: float=0.01,
                           sample_patients: pd.DataFrame=None):
    if os.path.splitext(fname)[-1] == '.csv':
        data = pd.read_csv(os.path.join(data_path, fname))
    elif os.path.splitext(fname)[-1] == '.parquet':
        data = pd.read_parquet(os.path.join(data_path, fname))
    else:
        raise ValueError('Unknown file format')

    original_dtypes = get_column_dtype(data)
    key_cols = [col for col in data.columns if col in cfg.preprocess.key_cols]
    time_cols = cfg.preprocess.time_cols

    if cfg.dataset.debug:
        if sample_patients is not None:
            data = data.merge(sample_patients[key_cols], on=key_cols)
        else:
            data = data.sample(cfg.dataset.debug_n)

    if excluded_fname is not None:
        with open(os.path.join(cfg.path.excluded_feature_path, excluded_fname), 'r') as f:
            excluded_features = [col.strip() for col in f.readlines()]
        data = data.drop(columns=excluded_features)

    for col in drug_name_cols:
        data[f'{col}_{value_col}'] = data.apply(lambda x: x[col] + '_' + x[value_col] if not pd.isnull(x[col]) else float('nan'), axis=1)

    drug_count = []
    for col in drug_name_cols:
        drug_count.append(data[f'{col}_{value_col}'].value_counts())
    drug_count = pd.concat(drug_count)
    drug_count = drug_count.groupby(level=0).sum()

    # drop columns that have not enough rows
    _data = []
    if row_threshold_ratio is not None:
        row_threshold = drug_count.sum() * row_threshold_ratio
        keep_drug = drug_count[drug_count >= row_threshold]
        for col in drug_name_cols:
            _data.append(data[data[f'{col}_{value_col}'].isin(keep_drug.index)])
        data = pd.concat(_data)
    data = data.drop_duplicates(subset=key_cols+time_cols)

    pivot_data = []
    for col in drug_name_cols:
        _data = data[key_cols+time_cols+[f'{col}_{value_col}']].copy()
        _data = _data.dropna(subset=[f'{col}_{value_col}']).reset_index(drop=True)

        _data = pd.pivot_table(_data, index=key_cols + time_cols, columns=f'{col}_{value_col}',
                               aggfunc=lambda x: 1 if len(x) > 0 else 0, fill_value=0)

        pivot_data.append(_data)
    pivot_data = pd.concat(pivot_data, axis=1)
    pivot_data = pivot_data.groupby(level=0, axis=1).first()
    pivot_data = pivot_data.fillna(0)

    feature_type = check_column_type(pivot_data)
    mask = pivot_data.map(lambda x: 0 if pd.isnull(x) else 1)

    if outliers:
        pivot_data, mask = remove_outliers(pivot_data, mask, outlier_threshold)

    if imputation:
        pivot_data = impute_missing_values(data=pivot_data,
                                           key_cols=key_cols,
                                           imputation_agg_func=imputation_agg_func,
                                           feature_type=feature_type)

    pivot_data = pivot_data.reset_index()
    mask = mask.reset_index()

    return pivot_data, mask, feature_type, key_cols, original_dtypes


@measure_runtime
def preprocess_diagnosis(cfg,
                         data_path: str,
                         fname: str='diagnosis.csv.parquet',
                         excluded_fname: str=None,
                         imputation: bool=True,
                         imputation_agg_func: str='mean',
                         outliers: bool=True,
                         outlier_threshold: dict=None,
                         row_threshold_ratio: float=0.01,
                         sample_patients: pd.DataFrame=None):
    if os.path.splitext(fname)[-1] == '.csv':
        data = pd.read_csv(os.path.join(data_path, fname))
    elif os.path.splitext(fname)[-1] == '.parquet':
        data = pd.read_parquet(os.path.join(data_path, fname))
    else:
        raise ValueError('Unknown file format')

    original_dtypes = get_column_dtype(data)

    key_cols = [col for col in data.columns if col in cfg.preprocess.key_cols]
    time_cols = cfg.preprocess.time_cols

    if cfg.dataset.debug:
        if sample_patients is not None:
            data = data.merge(sample_patients[key_cols], on=key_cols)
        else:
            data = data.sample(cfg.dataset.debug_n)

    if excluded_fname is not None:
        with open(os.path.join(cfg.path.excluded_feature_path, excluded_fname), 'r') as f:
            excluded_features = [col.strip() for col in f.readlines()]
        data = data.drop(columns=excluded_features)

    # drop columns that have not enough rows
    if row_threshold_ratio is not None:
        row_threshold = data.shape[0] * row_threshold_ratio
        unique_item_count = data['icd10_cm'].value_counts()
        keep_item = unique_item_count[unique_item_count >= row_threshold]
        data = data[data['icd10_cm'].isin(keep_item.index)]

    data = pd.pivot_table(data, index=key_cols+time_cols, columns='icd10_cm',
                          aggfunc=lambda x: 1 if len(x) > 0 else 0, fill_value=0)
    feature_type = check_column_type(data)
    mask = data.map(lambda x: 0 if pd.isnull(x) else 1)

    if outliers:
        data, mask = remove_outliers(data, mask, outlier_threshold)

    if imputation:
        data = impute_missing_values(data=data,
                                     key_cols=key_cols,
                                     imputation_agg_func=imputation_agg_func,
                                     feature_type=feature_type)

    data = data.reset_index()
    mask = mask.reset_index()

    return data, mask, feature_type, key_cols, original_dtypes


def split_dataset(train_operations: pd.DataFrame,
                  val_operations: pd.DataFrame,
                  test_operations: pd.DataFrame,
                  data: pd.DataFrame,
                  key_cols: List[str]):
    train_operations = train_operations[key_cols].drop_duplicates()
    val_operations = val_operations[key_cols].drop_duplicates()
    test_operations = test_operations[key_cols].drop_duplicates()

    train_data = pd.merge(train_operations, data, on=key_cols, how='inner')
    val_data = pd.merge(val_operations, data, on=key_cols, how='inner')
    test_data = pd.merge(test_operations, data, on=key_cols, how='inner')

    return train_data, val_data, test_data


def save_train_val_test(train_data: pd.DataFrame,
                        val_data: pd.DataFrame,
                        test_data: pd.DataFrame,
                        feature_type: dict,
                        hdf_key: str,
                        save_path: str,
                        save_fname: str='inspire.h5',
                        column_dtypes: dict=None,
                        feature_output_dimensions: dict=None):
    train_data.to_hdf(os.path.join(save_path, save_fname), key=f'{hdf_key}_train')
    val_data.to_hdf(os.path.join(save_path, save_fname), key=f'{hdf_key}_val')
    test_data.to_hdf(os.path.join(save_path, save_fname), key=f'{hdf_key}_test')

    # save meta data (feature type)
    modes = ['train', 'val', 'test']
    for mode in modes:
        with h5py.File(os.path.join(save_path, save_fname), 'a') as f:
            dataset = f[f'{hdf_key}_{mode}']
            for k, v in feature_type.items():
                dataset.attrs[f'feature_type_{k}'] = v

            if column_dtypes is not None:
                for k, v in column_dtypes.items():
                    dataset.attrs[f'column_dtype_{k}'] = str(v)

            if feature_output_dimensions is not None:
                for k, v in feature_output_dimensions.items():
                    dataset.attrs[f'feature_output_dimension_{k}'] = v


def match_id(row: pd.DataFrame,
             data: pd.DataFrame,
             patient_id: str='subject_id',
             start_time: str='admission_time',
             end_time: str='discharge_time',
             chart_time: str='chart_time',
             target_id: str='op_id'):
    if (target_id in row.index) and (not np.isnan(row[target_id])):
        return row[target_id]

    else:
        subject_data = data[data[patient_id] == row[patient_id]]
        for _, r in subject_data.iterrows():
            if r[start_time] <= row[chart_time] <= r[end_time]:
                return r[target_id]
        return None


def match_id_train_val_test(operations: pd.DataFrame,
                            train_data: pd.DataFrame,
                            val_data: pd.DataFrame,
                            test_data: pd.DataFrame,
                            patient_id: str='subject_id',
                            start_time: str='admission_time',
                            end_time: str='discharge_time',
                            chart_time: str='chart_time',
                            target_id: str='op_id'):
    train_data[target_id] = train_data.apply(lambda row: match_id(
        row, operations,
        patient_id=patient_id,
        start_time=start_time,
        end_time=end_time,
        chart_time=chart_time,
        target_id=target_id
    ), axis=1)
    val_data[target_id] = val_data.apply(lambda row: match_id(
        row, operations,
        patient_id=patient_id,
        start_time=start_time,
        end_time=end_time,
        chart_time=chart_time,
        target_id=target_id
    ), axis=1)
    test_data[target_id] = test_data.apply(lambda row: match_id(
        row, operations,
        patient_id=patient_id,
        start_time=start_time,
        end_time=end_time,
        chart_time=chart_time,
        target_id=target_id
    ), axis=1)

    return train_data, val_data, test_data


def merge_tables(data: List[pd.DataFrame],
                 key_cols: List[List[str]],
                 table_name: List[str]=None,
                 how: str='outer'):
    idx = 0
    df, left_key = None, None
    for _data, _key in zip(data, key_cols):
        if idx == 0:
            df = _data
        else:
            suffix = f'__{table_name[idx]}' if table_name is not None else '__'
            df = pd.merge(df, _data,
                          on=list(set(left_key).intersection(_key)),
                          how=how,
                          suffixes=('', suffix))

        left_key = _key
        idx += 1

    return df


def merge_type_dict(types: List[dict],
                    table_name: List[str]=None):
    idx = 0
    type_dict = dict()
    for _type in types:
        if table_name is not None:
            _dict = {f'{k}__{table_name[idx]}' if k in type_dict else k: v
                     for k, v in _type.items()}
        else:
            _dict = {f'{k}__' if k in type_dict else k: v
                     for k, v in _type.items()}
        idx += 1
        type_dict.update(_dict)

    return type_dict


def convert_type_by_feature_type(data: pd.DataFrame,
                                 feature_type: dict):
    for col, _type in feature_type.items():
        if _type == 'Binary':
            data[col] = data[col].astype(int)
        elif _type == 'Categorical':
            data[col] = data[col].astype(str)
        elif _type == 'Numerical':
            data[col] = data[col].astype(float)

    return data


def aggregate_data(data: pd.DataFrame, key_cols: List[str], feature_type: dict, group_cols: List[str],
                   time_cols: List[str]=None):
    agg_methods = dict()
    for col, _type in feature_type.items():
        if _type == 'Binary':
            agg_methods[col] = 'max'
        elif _type == 'Categorical':
            agg_methods[col] = 'first'
        elif _type == 'Numerical':
            agg_methods[col] = 'mean'
        else:
            raise ValueError(f'Unknown feature type: {_type}')
    drop_key = list(set(key_cols).difference(group_cols))
    data = data.drop(columns=drop_key)

    # group_cols should be in key_cols
    group_cols = list(set(group_cols).intersection(key_cols))
    if time_cols is not None:
        data = data.groupby(group_cols+time_cols, dropna=False).agg(agg_methods)
    else:
        data = data.groupby(group_cols, dropna=False).agg(agg_methods)
    data = data.reset_index()

    return data


def preprocess_categorical_features(data: pd.DataFrame,
                                    cols: List[str],
                                    max_num_categories: int=6,
                                    converted_categories: dict=None):
    if converted_categories is not None:
        for col, keep_categories in converted_categories.items():
            if col in cols:
                data[col] = data[col].apply(lambda x: x if x in keep_categories else 'Others')
    else:
        converted_categories = dict()
        for col in cols:
            num_categories = data[col].nunique()

            if num_categories >= max_num_categories:
                value_counts = data[col].value_counts()
                keep_categories = value_counts.index[:max_num_categories-1]
                data[col] = data[col].apply(lambda x: x if x in keep_categories else 'Others')
                converted_categories[col] = keep_categories

    return data, converted_categories


def sample_timepoints(data: pd.DataFrame,
                      time_info_data: pd.DataFrame,
                      col: str,
                      start_time_col: str,
                      end_time_col: str,
                      timepoints: int,
                      patient_id: str = 'subject_id',
                      op_id: str = 'op_id'):
    merged_data = data.merge(time_info_data[[patient_id, op_id, start_time_col, end_time_col]], on=patient_id)
    filtered_data = merged_data[(merged_data[col] >= merged_data[start_time_col]) &
                                (merged_data[col] < merged_data[end_time_col])]

    filtered_data['rank'] = filtered_data.groupby([patient_id, op_id])[col].rank(method='first')
    sampled_data = filtered_data[filtered_data['rank'] <= timepoints].drop(columns='rank')

    return sampled_data[[patient_id, op_id, col]]


def remove_outliers(data: pd.DataFrame,
                    mask_data: pd.DataFrame,
                    thresholds: dict):
    for col in data.columns:
        if col in thresholds:
            lower, upper = thresholds[col]['lower_bound'], thresholds[col]['upper_bound']
            indices = (data[col] < lower) | (data[col] > upper)
            data.loc[indices, col] = float('nan')
            mask_data.loc[indices, col] = 0

    return data, mask_data


def clip_data_by_timepoints(data: pd.DataFrame,
                            timepoints: int,
                            group_cols: str='subject_id',
                            padding: bool=False,
                            parallel: bool=True):
    def process_group(group):
        if len(group) >= timepoints:
            return group.iloc[:timepoints]

        elif padding:
            pad_rows = timepoints - len(group)
            pad_df = pd.DataFrame(
                {col: [float('nan')] * pad_rows for col in group.columns},
                index=[group.index[0]] * pad_rows
            )
            return pd.concat([group, pad_df])
        else:
            return group

    if parallel:
        pandarallel.initialize(progress_bar=True)
        clipped_data = data.groupby(group_cols, group_keys=False).parallel_apply(process_group)
    else:
        clipped_data = data.groupby(group_cols, group_keys=False).apply(process_group)

    clipped_data[group_cols] = clipped_data[group_cols].ffill()

    return clipped_data


if __name__ == "__main__":
    feature_types = [
        {'gender': 'Categorical', 'hh': 'Numerical', 'spo2': 'Numerical'},
        {'gender': 'Categorical', 'hh': 'Numerical', 'bt': 'Numerical'}
    ]
    feature_type = merge_type_dict(feature_types, table_name=['v', 'wv'])
    breakpoint()