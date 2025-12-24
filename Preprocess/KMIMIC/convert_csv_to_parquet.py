import re
import pandas as pd

from Utils.file import *
from Preprocess.utils import *


def get_object_dtype(data: pd.DataFrame):
    object_cols = data.select_dtypes(include=['object']).columns

    return object_cols


def convert_object_dtype(data: pd.DataFrame, object_cols: list):
    for col in object_cols:
        data[col] = data[col].astype(str)

    return data


def convert_csv_to_parquet(fpath: str='K_MIMIC'):
    import config_manager

    config_manager.load_config()
    cfg = config_manager.config

    data_path = cfg.path.raw_data_path
    fpath = os.path.join(data_path, fpath)

    files = get_all_files_recursive(fpath, ext='.csv')
    # table_list = [os.path.basename(file) for file in get_all_files_recursive(fpath, ext='.csv')]

    for file in files:
        table_name = os.path.basename(file).split('.')[0]
        save_fpath = os.path.dirname(file)
        save_fname = f'{table_name}.parquet'
        save_fpath = os.path.join(save_fpath, save_fname)

        if os.path.exists(save_fpath):
            print('Already exists:', table_name)
            continue

        try:
            data = pd.read_csv(file)
            object_cols = get_object_dtype(data)
            data = convert_object_dtype(data, object_cols)
            data.to_parquet(save_fpath)
            print('Saved:', table_name)

        except Exception as e:
            print('!!! Error:', table_name)
            print(e)


if __name__ == "__main__":
    convert_csv_to_parquet(fpath='K_MIMIC')