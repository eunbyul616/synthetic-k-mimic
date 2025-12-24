import os
import pandas as pd
import numpy as np
from time import time
import random
import itertools
from tqdm import tqdm


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


def classify_variables_types(data: pd.Series or np.array,
                             category_threshold: int=5) -> str:
    """
    Classify variables into numerical, categorical, binary, or other.

    Args:
        data: data to classify the type of variables
        category_threshold: threshold to determine if a numerical variable is categorical or not. Default is 5.

    Returns:
        type of variable. It can be 'Numerical', 'Categorical', 'Binary', or 'Other'.

    """

    if pd.api.types.is_numeric_dtype(data):
        if data.isin([0, 1]).all():
            return 'Binary'
        else:
            if data.apply(lambda x: float(x).is_integer()).all() and (data.nunique() <= category_threshold):
                return 'Categorical'
            else:
                return 'Numerical'
    elif pd.api.types.is_object_dtype(data):
        return 'Categorical'
    elif pd.api.types.is_bool_dtype(data):
        return 'Binary'
    else:
        return 'Other'
