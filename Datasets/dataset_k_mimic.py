from pathlib import Path
import numpy as np
from omegaconf import DictConfig
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader

from Utils.file import *
from Utils.dataset import *
from Utils.reproducibility import *
from Datasets.utils import *
from Manipulation.manipulation import Manipulation


class KMIMICDataset(Dataset):
    def __init__(self,
                 cfg: DictConfig,
                 dataset_name: str='K_MIMIC',
                 dataset_fname: str=None,
                 static_hdf5_key: str='static',
                 temporal_hdf5_key: str='temporal',
                 mode: str='train',
                 condition_col: list=None,
                 static_cols: List[str]=None,
                 temporal_cols: List[str]=None):
        assert mode in ['train', 'val', 'test'], 'Invalid mode'

        self.cfg = cfg
        self.verbose = cfg.dataset.verbose
        self.dataset_name = dataset_name

        self.condition_col = condition_col

        key_cols = cfg.data.key_cols
        time_cols = cfg.data.time_cols
        static_exclude_cols = cfg.data.excluded_cols.static
        temporal_exclude_cols = cfg.data.excluded_cols.temporal
        seq_len = cfg.dataloader.seq_len

        # prefix
        diagnoses_prefix = cfg.preprocess.icd_code.diagnoses_prefix
        procedure_prefix = cfg.preprocess.icd_code.procedure_prefix
        proc_prefix = cfg.preprocess.proc_prefix

        data_path = os.path.join(cfg.path.preprocessed_data_path, dataset_name)
        if dataset_fname is None:
            dataset_fname = 'K_MIMIC_preprocessed.h5'

        os.makedirs(os.path.join(cfg.path.transformer_path, dataset_name), exist_ok=True)

        transformer_name = f'{Path(dataset_fname).stem}_{cfg.manipulation.transformer_name}'

        static_transformer_fpath = os.path.join(cfg.path.transformer_path,
                                                dataset_name,
                                                f'{transformer_name}_static_transformer.pkl')
        static_transformed_data_fpath = os.path.join(cfg.path.transformer_path,
                                                     dataset_name,
                                                     f'{transformer_name}_static_{mode}.pkl')
        static_min_max_fpath = os.path.join(cfg.path.transformer_path,
                                            dataset_name,
                                            f'{transformer_name}_static_min_max_values.pkl')
        temporal_transformer_fpath = os.path.join(cfg.path.transformer_path,
                                                  dataset_name,
                                                  f'{transformer_name}_temporal_transformer.pkl')
        temporal_transformed_data_fpath = os.path.join(cfg.path.transformer_path,
                                                       dataset_name,
                                                       f'{transformer_name}_temporal_{mode}.pkl')
        temporal_min_max_fpath = os.path.join(cfg.path.transformer_path,
                                              dataset_name,
                                              f'{transformer_name}_temporal_min_max_values.pkl')

        static_data, static_type, static_output_dims = load_data(os.path.join(data_path, dataset_fname), static_hdf5_key, mode)
        temporal_data, temporal_type, temporal_output_dims = load_data(os.path.join(data_path, dataset_fname), temporal_hdf5_key, mode)
        static_data_key_cols = [col for col in key_cols if col in static_data.columns]
        temporal_data_key_cols = [col for col in key_cols if col in temporal_data.columns]

        static_data = static_data.set_index(static_data_key_cols)
        temporal_data = temporal_data.set_index(temporal_data_key_cols + time_cols)

        # define Manipulation
        if mode == 'train':
            # condition
            condition = static_data[self.condition_col] if self.condition_col is not None else None
            if condition is not None:
                static_exclude_cols += self.condition_col

                icu_expire_flag = torch.tensor(condition['icu_expire_flag'].values)
                hospital_expire_flag = torch.tensor(condition['hospital_expire_flag'].values)
                labels = torch.where(icu_expire_flag == 1, 1, torch.where(hospital_expire_flag == 1, 2, 0))
                self.condition = torch.nn.functional.one_hot(labels, num_classes=3).float()
            else:
                self.condition = None

            # static
            static_exclude_cols += [col for col in static_data.columns if static_data[col].nunique() == 1]
            if len(static_exclude_cols) > 0:
                static_data = static_data.drop(columns=static_exclude_cols)

            num_cols = [col for col in static_data.columns if static_type[col] == 'Numerical']
            min_max_values = static_data[num_cols].agg(['min', 'max']).to_dict()
            save_pkl(min_max_values, static_min_max_fpath)

            static_transformer = Manipulation(
                verbose=self.verbose,
                numerical_activation_fn=cfg.manipulation.activation_fn.numerical,
                categorical_activation_fn=cfg.manipulation.activation_fn.categorical,
                binary_activation_fn=cfg.manipulation.activation_fn.binary,
                min_max_values=min_max_values,
                drop_first=cfg.manipulation.drop_first,
            )

            # temporal
            temporal_exclude_cols += [col for col in temporal_data.columns if temporal_data[col].nunique() == 1]
            if len(temporal_exclude_cols) > 0:
                temporal_data = temporal_data.drop(columns=temporal_exclude_cols)

            num_cols = [col for col in temporal_data.columns if temporal_type[col] == 'Numerical']
            min_max_values = temporal_data[num_cols].agg(['min', 'max']).to_dict()
            save_pkl(min_max_values, temporal_min_max_fpath)

            temporal_transformer = Manipulation(
                verbose=self.verbose,
                numerical_activation_fn=cfg.manipulation.activation_fn.numerical,
                categorical_activation_fn=cfg.manipulation.activation_fn.categorical,
                binary_activation_fn=cfg.manipulation.activation_fn.binary,
                min_max_values=min_max_values,
                drop_first=cfg.manipulation.drop_first,
            )
        else:
            # condition
            condition = static_data[self.condition_col] if self.condition_col is not None else None
            if condition is not None:
                static_exclude_cols += self.condition_col

                icu_expire_flag = torch.tensor(condition['icu_expire_flag'].values)
                hospital_expire_flag = torch.tensor(condition['hospital_expire_flag'].values)
                labels = torch.where(icu_expire_flag == 1, 1, torch.where(hospital_expire_flag == 1, 2, 0))
                self.condition = torch.nn.functional.one_hot(labels, num_classes=3).float()
            else:
                self.condition = None

            # static
            min_max_values = load_pkl(static_min_max_fpath)
            static_transformer = Manipulation(
                verbose=self.verbose,
                numerical_activation_fn=cfg.manipulation.activation_fn.numerical,
                categorical_activation_fn=cfg.manipulation.activation_fn.categorical,
                binary_activation_fn=cfg.manipulation.activation_fn.binary,
                min_max_values=min_max_values,
                drop_first=cfg.manipulation.drop_first,
            )
            if os.path.exists(static_transformer_fpath):
                print('Transformer found. Load transformer.')
                static_transformer = static_transformer.load(static_transformer_fpath)

            icd_d_cols = [col for col in static_data.columns if diagnoses_prefix in col]
            icd_p_cols = [col for col in static_data.columns if procedure_prefix in col]
            include_cols = ([col.column_name for col in static_transformer._data_manipulation_info_list
                            if ('mask' not in col.column_name) and (not col.column_name.startswith((diagnoses_prefix, procedure_prefix)))] +
                            icd_d_cols + icd_p_cols)
            static_data = static_data[include_cols]

            # temporal
            min_max_values = load_pkl(temporal_min_max_fpath)
            temporal_transformer = Manipulation(
                verbose=self.verbose,
                numerical_activation_fn=cfg.manipulation.activation_fn.numerical,
                categorical_activation_fn=cfg.manipulation.activation_fn.categorical,
                binary_activation_fn=cfg.manipulation.activation_fn.binary,
                min_max_values=min_max_values,
                drop_first=cfg.manipulation.drop_first,
            )
            if os.path.exists(temporal_transformer_fpath):
                print('Transformer found. Load transformer.')
                temporal_transformer = temporal_transformer.load(temporal_transformer_fpath)

            proc_cols = [col for col in temporal_data.columns if proc_prefix in col]
            include_cols = [col.column_name for col in temporal_transformer._data_manipulation_info_list
                            if ('mask' not in col.column_name) and (not col.column_name.startswith(proc_prefix))] + proc_cols
            temporal_data = temporal_data[include_cols]

        # mask
        if cfg.dataloader.mask:
            static_mask_data, _, _ = load_data(os.path.join(data_path, dataset_fname), f'{static_hdf5_key}_mask', mode)
            temporal_mask_data, _, _ = load_data(os.path.join(data_path, dataset_fname), f'{temporal_hdf5_key}_mask', mode)
            static_mask_data = static_mask_data.set_index(static_data_key_cols)
            temporal_mask_data = temporal_mask_data.set_index(temporal_data_key_cols + time_cols)

            if mode == 'train':
                if len(static_exclude_cols) > 0:
                    static_mask_data = static_mask_data.drop(columns=static_exclude_cols)
                if len(temporal_exclude_cols) > 0:
                    temporal_mask_data = temporal_mask_data.drop(columns=temporal_exclude_cols)
            else:
                icd_d_cols = [col for col in static_data.columns if diagnoses_prefix in col]
                icd_p_cols = [col for col in static_data.columns if procedure_prefix in col]
                include_cols = ([col.column_name for col in static_transformer._data_manipulation_info_list
                            if ('mask' not in col.column_name) and (not col.column_name.startswith((diagnoses_prefix, procedure_prefix)))] +
                            icd_d_cols + icd_p_cols)
                static_mask_data = static_mask_data[include_cols]

                proc_cols = [col for col in temporal_data.columns if proc_prefix in col]
                include_cols = [col.column_name for col in temporal_transformer._data_manipulation_info_list
                                if ('mask' not in col.column_name) and (not col.column_name.startswith(proc_prefix))] + proc_cols
                temporal_mask_data = temporal_mask_data[include_cols]

        if static_cols is None:
            static_type = {col: static_type[col] for col in static_data.columns
                           if col not in static_exclude_cols}
        else:
            static_type = {col: static_type[col] for col in static_data.columns
                           if (col in static_cols) and (col not in static_exclude_cols)}

        if temporal_cols is None:
            temporal_type = {col: temporal_type[col] for col in temporal_data.columns
                             if col not in temporal_exclude_cols}
        else:
            temporal_type = {col: temporal_type[col] for col in temporal_data.columns
                             if (col in temporal_cols) and (col not in temporal_exclude_cols)}

        s_cols = static_data.columns
        t_cols = temporal_data.columns

        # process icd_d, icd_p (listwise binary columns)
        icd_d_cols = [col for col in s_cols if diagnoses_prefix in col]
        icd_p_cols = [col for col in s_cols if procedure_prefix in col]

        s_cols = list(set(s_cols) - set(icd_d_cols) - set(icd_p_cols))
        s_cols = s_cols + [diagnoses_prefix, procedure_prefix]
        static_mask_data[diagnoses_prefix] = static_mask_data[icd_d_cols].max(axis=1)
        static_mask_data[procedure_prefix] = static_mask_data[icd_p_cols].max(axis=1)
        static_mask_data = static_mask_data.drop(columns=icd_d_cols + icd_p_cols)

        static_data = static_data[s_cols[:-2] + icd_p_cols + icd_d_cols]
        static_mask_data = static_mask_data[s_cols]

        proc_cols = [col for col in t_cols if proc_prefix in col]
        t_cols = list(set(t_cols) - set(proc_cols))
        t_cols = t_cols + [proc_prefix]
        temporal_mask_data[proc_prefix] = temporal_mask_data[proc_cols].max(axis=1)
        temporal_mask_data = temporal_mask_data.drop(columns=proc_cols)

        temporal_data = temporal_data[t_cols[:-1] + proc_cols]
        temporal_mask_data = temporal_mask_data[t_cols]

        if cfg.dataloader.mask:
            if static_cols is None:
                static_mask_type = {f'{col}_mask': 'Binary' for col in static_mask_data.columns
                                    if col not in static_exclude_cols}
            else:
                static_mask_type = {f'{col}_mask': 'Binary' for col in static_mask_data.columns
                                    if (col in static_cols) and (col not in static_exclude_cols)}
            if temporal_cols is None:
                temporal_mask_type = {f'{col}_mask': 'Binary' for col in temporal_mask_data.columns
                                      if col not in temporal_exclude_cols}
            else:

                temporal_mask_type = {f'{col}_mask': 'Binary' for col in temporal_mask_data.columns
                                      if (col in temporal_cols) and (col not in temporal_exclude_cols)}

            static_type.update(static_mask_type)
            temporal_type.update(temporal_mask_type)

            # temporal_data = temporal_data[t_cols]
            # temporal_mask_data = temporal_mask_data[t_cols]

            sm_cols = [f'{col}_mask' for col in s_cols]
            tm_cols = [f'{col}_mask' for col in t_cols]

            static_mask_data.columns = sm_cols
            temporal_mask_data.columns = tm_cols

            static_data = pd.concat([static_data, static_mask_data], axis=1)
            temporal_data = pd.concat([temporal_data, temporal_mask_data], axis=1)

        temporal_data = temporal_data.reset_index()
        padding_idx = temporal_data[time_cols].isna().any(axis=1)

        # forward fill
        if not cfg.dataloader.pad_flag:
            temporal_data[t_cols] = temporal_data[t_cols].fillna(method='ffill')
            temporal_data[tm_cols] = temporal_data[tm_cols].fillna(0)
            temporal_data[time_cols] = temporal_data[time_cols].fillna(method='ffill')
        temporal_data = temporal_data.set_index(temporal_data_key_cols + time_cols)

        static_transformer, static_transformed_data = manipulate_data(cfg=cfg,
                                                                      transformer_path=static_transformer_fpath,
                                                                      transformed_data_path=static_transformed_data_fpath,
                                                                      transformer=static_transformer,
                                                                      data=static_data,
                                                                      feature_type=static_type)
        (sc_data, sn_data, sc_mask_data, sn_mask_data, sc_cols, sn_cols,
         sl_cols, sl_data, sl_mask_data) = split_data_feature_type(
            static_transformed_data,
            static_type,
            static_transformer,
            listwise_feature_prefix=[diagnoses_prefix, procedure_prefix]
        )

        temporal_transformer, temporal_transformed_data = manipulate_data(cfg=cfg,
                                                                          transformer_path=temporal_transformer_fpath,
                                                                          transformed_data_path=temporal_transformed_data_fpath,
                                                                          transformer=temporal_transformer,
                                                                          data=temporal_data,
                                                                          feature_type=temporal_type)
        if cfg.dataloader.pad_flag:
            temporal_transformed_data[padding_idx] = cfg.dataloader.pad_value

        (tc_data, tn_data, tc_mask_data, tn_mask_data, tc_cols, tn_cols,
         tl_cols, tl_data, tl_mask_data) = split_data_feature_type(
            temporal_transformed_data,
            temporal_type,
            temporal_transformer,
            listwise_feature_prefix=[proc_prefix]
        )

        tc_data = tc_data.reshape(-1, seq_len, tc_data.shape[-1])
        tn_data = tn_data.reshape(-1, seq_len, tn_data.shape[-1])
        tl_data = tl_data.reshape(-1, seq_len, tl_data.shape[-1])

        tc_mask_data = tc_mask_data.reshape(-1, seq_len, tc_mask_data.shape[-1])
        tn_mask_data = tn_mask_data.reshape(-1, seq_len, tn_mask_data.shape[-1])
        tl_mask_data = tl_mask_data.reshape(-1, seq_len, tl_mask_data.shape[-1])

        self.static_data = static_data
        self.temporal_data = temporal_data

        self.static_type = static_type
        self.temporal_type = temporal_type

        self.static_transformer = static_transformer
        self.temporal_transformer = temporal_transformer

        self.sc_data = torch.tensor(sc_data, dtype=torch.float32)
        self.sc_cols = sc_cols
        self.sl_data = torch.tensor(sl_data, dtype=torch.float32)
        self.sl_cols = sl_cols

        self.tc_data = torch.tensor(tc_data, dtype=torch.float32)
        self.tc_cols = tc_cols
        self.tl_data = torch.tensor(tl_data, dtype=torch.float32)
        self.tl_cols = tl_cols

        self.sn_data = torch.tensor(sn_data, dtype=torch.float32)
        self.sn_cols = sn_cols
        self.tn_data = torch.tensor(tn_data, dtype=torch.float32)
        self.tn_cols = tn_cols
        self.sc_mask_data = torch.tensor(sc_mask_data, dtype=torch.float32)
        self.tc_mask_data = torch.tensor(tc_mask_data, dtype=torch.float32)
        self.sn_mask_data = torch.tensor(sn_mask_data, dtype=torch.float32)
        self.tn_mask_data = torch.tensor(tn_mask_data, dtype=torch.float32)
        self.sl_mask_data = torch.tensor(sl_mask_data, dtype=torch.float32)
        self.tl_mask_data = torch.tensor(tl_mask_data, dtype=torch.float32)

        del static_data, temporal_data, static_mask_data, temporal_mask_data
        del sc_data, tc_data, sc_mask_data, tc_mask_data
        del sn_data, tn_data, sn_mask_data, tn_mask_data
        del static_transformed_data, temporal_transformed_data
        del static_transformer, temporal_transformer

    def __getitem__(self, idx):
        static_mask_data = torch.concatenate([self.sc_mask_data[idx], self.sl_mask_data[idx], self.sn_mask_data[idx]], dim=-1)
        temporal_mask_data = torch.concatenate([self.tc_mask_data[idx], self.tl_mask_data[idx], self.tn_mask_data[idx]], dim=-1)

        if self.condition_col is not None:

            return (self.sc_data[idx], self.tc_data[idx],
                    self.sn_data[idx], self.tn_data[idx],
                    self.sl_data[idx], self.tl_data[idx],
                    static_mask_data, temporal_mask_data,
                    self.condition[idx])
        else:
            return (self.sc_data[idx], self.tc_data[idx],
                    self.sn_data[idx], self.tn_data[idx],
                    self.sl_data[idx], self.tl_data[idx],
                    static_mask_data, temporal_mask_data)

    def __len__(self):
        return len(self.sc_data)

    @property
    def static_key_df(self):
        return pd.DataFrame(list(self.static_data.index), columns=self.cfg.data.key_cols)

    @property
    def temporal_key_df(self):
        return pd.DataFrame(list(self.temporal_data.index), columns=self.cfg.data.key_cols + self.cfg.data.time_cols)


if __name__ == "__main__":
    import config_manager

    config_manager.load_config()
    cfg = config_manager.config
    cols = None

    dataset_name = cfg.dataset.dataset_name
    fname = cfg.preprocess.preprocess_fname_suffix
    train_ratio, test_ratio = cfg.preprocess.train_valid_test_split_ratio
    dataset_fname = f'{dataset_name}_{fname}_{int(train_ratio * 10)}.h5'

    train_dataset = KMIMICDataset(cfg=cfg,
                                  dataset_name=dataset_name,
                                  dataset_fname=dataset_fname,
                                  mode='train',
                                  condition_col=getattr(cfg.data, 'condition_col', None),
                                  static_cols=cols)
    validation_dataset = KMIMICDataset(cfg=cfg,
                                       dataset_name=dataset_name,
                                       dataset_fname=dataset_fname,
                                       mode='val',
                                       condition_col=getattr(cfg.data, 'condition_col', None),
                                       static_cols=cols)
    test_dataset = KMIMICDataset(cfg=cfg,
                                 dataset_name=dataset_name,
                                 dataset_fname=dataset_fname,
                                 mode='test',
                                 condition_col=getattr(cfg.data, 'condition_col', None),
                                 static_cols=cols)

