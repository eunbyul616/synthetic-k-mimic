import pandas as pd
import re


def transform_race(data: pd.Series) -> pd.Series:
    """
    Transform values in the 'race' column to one of the following categories
    : 'ASIAN', 'WHITE', 'BLACK', 'HISPANIC/LATINO', 'UNKNOWN', 'OTHER'.
    Args:
        data: data to be transformed. race column

    Returns:
        pd.Series: transformed data
    """
    def convert_to_unknown(x):
        x = str(x)
        if x in ['UNABLE TO OBTAIN', 'UNKNOWN']:
            return 'UNKNOWN'
        else:
            return x

    def convert_to_asian(x):
        x = str(x)
        if re.search('ASIAN', x):
            return 'ASIAN'
        else:
            return x

    def convert_to_white(x):
        x = str(x)
        if re.search('WHITE', x):
            return 'WHITE'
        else:
            return x

    def convert_to_black(x):
        x = str(x)
        if re.search('BLACK', x):
            return 'BLACK'
        else:
            return x

    def convert_to_hispanic_or_latino(x):
        x = str(x)
        if re.search('HISPANIC/LATINO', x) or x == 'HISPANIC OR LATINO':
            return 'HISPANIC/LATINO'
        else:
            return x

    def convert_to_other(x):
        x = str(x)
        if x not in ['ASIAN', 'WHITE', 'BLACK', 'HISPANIC/LATINO', 'UNKNOWN']:
            return 'OTHER'
        else:
            return x

    convert_fns = {
        'ASIAN': convert_to_asian,
        'WHITE': convert_to_white,
        'BLACK': convert_to_black,
        'HISPANIC/LATINO': convert_to_hispanic_or_latino,
        'UNKNOWN': convert_to_unknown,
        'OTHER': convert_to_other
    }

    for cat, func in convert_fns.items():
        if cat != 'OTHER':
            data = data.apply(func)
    data = data.apply(convert_fns['OTHER'])

    return data


def transform_static_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform static data
    Args:
        data: static data

    Returns:
        pd.DataFrame: transformed static data
    """
    data['race'] = transform_race(data['race'])

    return data


def transform_temporal_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform temporal data
    Args:
        data: temporal data

    Returns:
        pd.DataFrame: transformed temporal data
    """
    # TBD
    return data

