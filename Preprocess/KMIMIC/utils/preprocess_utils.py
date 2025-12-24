import pandas as pd
from copy import deepcopy
from Preprocess.KMIMIC.utils.utils import extract_events_items_with_itemid


def pivot_events(data: pd.DataFrame, fillna: bool=False, value: int=None) -> pd.DataFrame:
    """
    Pivot events data
    Args:
        data: events data
        fillna: fill missing values (default: False)
        value: value to fill missing values (default: None)

    Returns:
        pd.DataFrame: pivoted events data
    """

    # error occurred when using pivot (ValueError: Index contains duplicate entries, cannot reshape)
    # data = data.pivot(index=['subject_id', 'hadm_id', 'stay_id', 'hours'], columns='label', values='value')

    data = data.pivot_table(index=['subject_id', 'hadm_id', 'stay_id', 'hours'], columns='label',
                            values='value', aggfunc='last')

    if fillna:
        data = data.fillna(value)
    data = data.reset_index()

    return data


def aggregate_events_by_label(data: pd.DataFrame,
                              itemid_map: dict,
                              key_cols: list,
                              time_cols: list=['hours'],
                              label_col: str='label',
                              value_col: str='value',
                              agg_func='mean',
                              keep: bool=False,
                              verbose: bool=False) -> pd.DataFrame:
    """
    Aggregate events by label
    Args:
        data: events data
        itemid_map: itemid map
        key_cols: key columns
        time_cols: time columns (default: ['hours'])
        label_col: item label column (default: 'label')
        value_col: measurement value column (default: 'value')
        agg_func: aggregation function (default: 'mean')
        keep: keep the original columns (default: False)
        verbose: verbose mode (default: False)

    Returns:

    """
    # convert label to LEVEL2 label for aggregation
    data = extract_events_items_with_itemid(data, itemid_map, keep=keep, verbose=verbose)
    data = data.groupby(key_cols+time_cols+[label_col])[value_col].agg(agg_func).reset_index()

    return data


def aggregate_data_by_hours(data: pd.DataFrame,
                            key_cols: list,
                            col: str='hours',
                            agg_func='median',
                            min_hours_value: int=0) -> pd.DataFrame:
    """"
    Aggregate data by hours

    Args:
        data: data
        key_cols: key columns
        col: column to aggregate
        agg_func: aggregation function
        min_hours_value: minimum hours value
    """
    data[col] = data[col].astype(int)

    if min_hours_value is not None:
        data[col] = data[col].apply(lambda x: min_hours_value if x <= min_hours_value else x)

    data = data.groupby(key_cols+[col]).agg(agg_func).reset_index()

    return data


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
    # linear interpolation
    _data = data.groupby(key_cols)[col].apply(lambda x: x.interpolate(method='linear',
                                                                      limit_direction='forward')).reset_index()
    _data = _data.groupby(key_cols)[col].apply(lambda x: x.ffill()).reset_index()
    _data = _data.groupby(key_cols)[col].apply(lambda x: x.bfill()).reset_index()

    if method == 'median':
        _median = data.groupby(key_cols)[col].median()
        _data = _data.fillna(_median)

        _total = data[col].median()
        _data = _data.fillna(_total)
        return _data[col]

    elif method == 'mean':
        _mean = data.groupby(key_cols)[col].mean()
        _data = _data.fillna(_mean)

        _total = data[col].mean()
        _data = _data.fillna(_total)

        return _data[col]
    else:
        raise ValueError('Invalid method')


def mask_missing_values(data: pd.DataFrame,
                        cols: list,
                        mask_col_pattern: str='_mask') -> pd.DataFrame:
    """"
    Mask missing values

    Args:
        data: data
        cols: columns to mask
        mask_col_pattern: mask column pattern
    """
    mask_data = deepcopy(data)
    for col in cols:
        mask_data[col] = (~data[col].isnull()).astype(float)
        mask_data.rename(columns={col: f'{col}{mask_col_pattern}'}, inplace=True)

    return mask_data


def filter_cohort_on_num_timestamps(data: pd.DataFrame,
                                    max_num_timestamps: int=30) -> pd.DataFrame:
    """
    Filter the cohort on the number of timestamps
    Args:
        data:
        max_num_timestamps:

    Returns:

    """
    _data = data.groupby(['subject_id', 'hadm_id', 'stay_id']).apply(lambda x: x.shape[0])
    _data = _data[_data <= max_num_timestamps].reset_index()
    data = data.merge(_data[['subject_id', 'hadm_id', 'stay_id']], on=['subject_id', 'hadm_id', 'stay_id'])

    return data


def impute_missing_hours(data: pd.DataFrame, max_num_timestamps: int=30) -> pd.DataFrame:
    """
    Impute missing hours in the data having less the number of timestamps than max_num_timestamps
    Args:
        data:
        max_num_timestamps:

    Returns:

    """
    _min_hours_by_stay = data.groupby(['subject_id', 'hadm_id', 'stay_id'])['hours'].min().reset_index()
    _max_hours_by_stay = data.groupby(['subject_id', 'hadm_id', 'stay_id'])['hours'].max().reset_index()
    _count_hours_by_stay = data.groupby(['subject_id', 'hadm_id', 'stay_id'])['hours'].count().reset_index()

    _stay = _min_hours_by_stay[['subject_id', 'hadm_id', 'stay_id']]

    _stay_df = []
    for i in range(_stay.shape[0]):
        _subject_id, _hadm_id, _stay_id = _stay.iloc[i][['subject_id', 'hadm_id', 'stay_id']].values

        _count = _count_hours_by_stay[(_count_hours_by_stay['subject_id'] == _subject_id)&
                                      (_count_hours_by_stay['hadm_id'] == _hadm_id)&
                                      (_count_hours_by_stay['stay_id'] == _stay_id)]['hours'].values[0]
        if _count < max_num_timestamps:
            _df = pd.DataFrame()
            _min = _min_hours_by_stay[(_min_hours_by_stay['subject_id'] == _subject_id)&
                                      (_min_hours_by_stay['hadm_id'] == _hadm_id)&
                                      (_min_hours_by_stay['stay_id'] == _stay_id)]['hours'].values[0]
            _max = _max_hours_by_stay[(_max_hours_by_stay['subject_id'] == _subject_id) &
                                      (_max_hours_by_stay['hadm_id'] == _hadm_id) &
                                      (_max_hours_by_stay['stay_id'] == _stay_id)]['hours'].values[0]

            _diff = max_num_timestamps - _count
            _df = pd.concat([_df, pd.Series([j for j in range(int(_max+1), int(_max+_diff+1))], name='hours')], axis=1)
            _df['subject_id'] = _subject_id
            _df['hadm_id'] = _hadm_id
            _df['stay_id'] = _stay_id

            _df = pd.concat([data[(data['subject_id'] == _subject_id)&
                                  (data['hadm_id'] == _hadm_id)&
                                  (data['stay_id'] == _stay_id)][['subject_id', 'hadm_id', 'stay_id', 'hours']],
                             _df[['subject_id', 'hadm_id', 'stay_id', 'hours']]], axis=0)

            _stay_df.append(_df)
        else:
            _df = data[(data['subject_id'] == _subject_id)&
                                   (data['hadm_id'] == _hadm_id)&
                                   (data['stay_id'] == _stay_id)][['subject_id', 'hadm_id', 'stay_id', 'hours']]
            _stay_df.append(_df)
    _stay_df = pd.concat(_stay_df)

    data = data.merge(_stay_df, on=['subject_id', 'hadm_id', 'stay_id', 'hours'], how='right')

    return data


def convert_data_unit(data: pd.DataFrame,
                      label_col: str='label',
                      unit_col: str='valueuom') -> pd.DataFrame:
    """"
    Convert data unit
    """

    unit_conversion_fn = {
        'weight': {
            'oz': lambda x: x / 16. * 0.45359237,
            'lbs': lambda x: x * 0.45359237},
        'height': {
            'in': lambda x: x * 2.54
        },
        'temperature': {
            'f': lambda x: (x - 32) * 5. / 9.
        }
    }

    for _type, fn_dict in unit_conversion_fn.items():
        for unit, fn in fn_dict.items():
            label_filter = data[label_col].str.contains(_type, case=False, na=False)
            unit_filter = (data[label_col].str.contains(unit, case=False, na=False) |
                           data[unit_col].str.contains(unit, case=False, na=False))

            data.loc[label_filter & unit_filter, 'value'] = data.loc[label_filter & unit_filter, 'value'].apply(fn)

    return data
