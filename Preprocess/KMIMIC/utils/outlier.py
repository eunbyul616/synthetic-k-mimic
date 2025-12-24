import numpy as np
import pandas as pd


def remove_outlier_on_percentile(data: pd.Series, lower_percentile: float=0.1, upper_percentile: float=99.9) -> (pd.Series, tuple):
    """
    Remove outliers on the given percentile range.
    Args:
        data: data to be processed
        lower_percentile: lower percentile
        upper_percentile: upper percentile

        Returns:

    """
    lower_bound = np.nanpercentile(data, lower_percentile, axis=0)
    upper_bound = np.nanpercentile(data, upper_percentile, axis=0)
    bound = (lower_bound, upper_bound)
    data[(data < lower_bound) | (data > upper_bound)] = np.nan

    return data, bound


def remove_outlier_by_limits(data: pd.Series,
                             lower_limit: float=None,
                             upper_limit: float=None,
                             include_lower: bool=None,
                             include_upper: bool=None) -> (pd.Series, tuple):

    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")

    if lower_limit is not None and upper_limit is not None and lower_limit > upper_limit:
        raise ValueError("Lower limit cannot be greater than upper limit.")

    filtered_data = data.copy()
    mask = pd.Series(True, index=filtered_data.index)

    if lower_limit is not None:
        mask &= filtered_data >= lower_limit if include_lower else filtered_data > lower_limit

    if upper_limit is not None:
        mask &= filtered_data <= upper_limit if include_upper else filtered_data < upper_limit

    filtered_data[~mask] = np.nan
    bound = (lower_limit, upper_limit, include_lower, include_upper)

    return filtered_data, bound