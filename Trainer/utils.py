import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List

import torch

from Visualization import *
from Manipulation.manipulation import Manipulation, DataManipulationInfo


def apply_activation(x_hat: List[torch.Tensor] or torch.Tensor,
                     feature_info: list,
                     logit_threshold: float = 0.5) -> torch.Tensor:
    if isinstance(x_hat, list):
        act_x_hat = []
        for i in range(len(x_hat)):
            if feature_info[i].column_type == 'Binary':
                _act_x_hat = torch.sigmoid(x_hat[i])
                _act_x_hat = (_act_x_hat >= logit_threshold).float()
                act_x_hat.append(_act_x_hat)
            elif feature_info[i].column_type == 'Categorical':
                act_x_hat.append(torch.softmax(x_hat[i], dim=-1))
            else:
                act_x_hat.append(torch.sigmoid(x_hat[i]))
        act_x_hat = torch.concatenate(act_x_hat, dim=-1)
    else:
        if feature_info[0].column_type == 'Binary':
            act_x_hat = torch.sigmoid(x_hat)
            act_x_hat = (act_x_hat >= logit_threshold).float()
        elif feature_info[0].column_type == 'Categorical':
            act_x_hat = torch.softmax(x_hat, dim=-1)
        else:
            # act_x_hat = torch.sigmoid(x_hat)
            act_x_hat = x_hat

    return act_x_hat


def apply_activation_static(x_hat: List[torch.Tensor] or torch.Tensor,
                            feature_info: list,
                            sc_cols=None,
                            sl_cols=None,
                            logit_threshold: float = 0.5) -> torch.Tensor:
    if isinstance(x_hat, list):
        act_x_hat = []
        for i, col in enumerate(sc_cols + sl_cols):
            dim = x_hat[i].shape[-1]

            if col in sc_cols:
                if feature_info[i].column_type == 'Binary':
                    _act_x_hat = torch.sigmoid(x_hat[i])
                    _act_x_hat = (_act_x_hat >= logit_threshold).float()
                    act_x_hat.append(_act_x_hat)
                elif feature_info[i].column_type == 'Categorical':
                    act_x_hat.append(torch.softmax(x_hat[i], dim=-1))
                elif feature_info[i].column_type == 'Listwise':
                    act_x_hat = torch.sigmoid(x_hat[i])
                    act_x_hat = (act_x_hat >= logit_threshold).float()
            else: # listwise
                _act_x_hat = torch.sigmoid(x_hat[i])
                _act_x_hat = (_act_x_hat >= logit_threshold).float()
                act_x_hat.append(_act_x_hat)
        act_x_hat = torch.concatenate(act_x_hat, dim=-1)
    else:
        if feature_info[0].column_type == 'Binary':
            act_x_hat = torch.sigmoid(x_hat)
            act_x_hat = (act_x_hat >= logit_threshold).float()
        elif feature_info[0].column_type == 'Categorical':
            act_x_hat = torch.softmax(x_hat, dim=-1)
        elif feature_info[0].column_type == 'Listwise':
            act_x_hat = torch.sigmoid(x_hat)
            act_x_hat = (act_x_hat >= logit_threshold).float()
        else:
            # act_x_hat = torch.sigmoid(x_hat)
            act_x_hat = x_hat

    return act_x_hat


def split_data_by_type(x_hat, sc_rep, tc_rep, sn, tn, sm, tm):
    s_idx = 0
    # static features
    rep_dim = sc_rep.shape[-1]
    dim = sc_rep.shape[-1] + sn.shape[-1]
    sc_rep_hat = x_hat[:, s_idx:s_idx + rep_dim]
    sn_hat = x_hat[:, s_idx + rep_dim:s_idx + dim]
    s_idx += dim

    # temporal features
    rep_dim = tc_rep.shape[-1]
    dim = tc_rep.shape[-1] + (tn.shape[1] * tn.shape[2])
    tc_rep_hat = x_hat[:, s_idx:s_idx + rep_dim]
    tn_hat = x_hat[:, s_idx + rep_dim:s_idx + dim]
    s_idx += dim

    # mask loss
    sm_dim = sm.shape[-1]
    dim = sm.shape[-1] + (tm.shape[1] * tm.shape[2])
    static_mask_hat = x_hat[:, s_idx:s_idx + sm_dim]
    temporal_mask_hat = x_hat[:, s_idx + sm_dim:s_idx + dim]
    s_idx += dim

    return sc_rep_hat, tc_rep_hat, sn_hat, tn_hat, static_mask_hat, temporal_mask_hat


def split_generated_data(x_hat, dims):
    out = []
    s_idx = 0
    for dim in dims:
        out.append(x_hat[:, s_idx:s_idx + dim])
        s_idx += dim

    return out


def eval_numerical_features(cols: List[str],
                            real: pd.DataFrame,
                            synthetic: pd.DataFrame,
                            labels: List[str]=['Real', 'Synthetic'],
                            save_path: str=None,
                            epoch: int=None) -> None:
    for col in cols:
        _col = col.replace('/', ' ').replace('>', ' ')
        fname = f'cdf_{_col}_epoch_{epoch}.png' if epoch is not None else f'cdf_{_col}.png'
        vis_cdf(data=[real[col], synthetic[col]],
                label=labels,
                title=col,
                save_path=os.path.join(save_path, fname) if save_path is not None else None)


def eval_categorical_features(cols: List[str],
                              real: pd.DataFrame,
                              synthetic: pd.DataFrame,
                              labels: List[str]=['Real', 'Synthetic'],
                              stat: str='percent',
                              save_path: str=None,
                              epoch: int=None) -> None:
    for col in cols:
        _col = col.replace('/', ' ').replace('>', ' ')
        fname = f'countplot_{_col}_epoch_{epoch}.png' if epoch is not None else f'countplot_{_col}.png'
        countplot_categorical_feature(data1=real,
                                      data2=synthetic,
                                      col=col,
                                      stat=stat,
                                      label1=labels[0],
                                      label2=labels[1],
                                      title=col,
                                      save_path=os.path.join(save_path, fname) if save_path is not None else None)


def inverse_transform(real: np.array,
                      synthetic: np.array,
                      transformer: Manipulation,
                      feature_info: List[DataManipulationInfo]=None,
                      mask: np.array=None) -> (pd.DataFrame, pd.DataFrame):
    if mask is None:
        real = transformer.inverse_transform(real, feature_info)
        synthetic = transformer.inverse_transform(synthetic, feature_info)
    else:
        real = np.where(mask, real, np.nan)
        synthetic = np.where(mask, synthetic, np.nan)

        real = transformer.inverse_transform(real, feature_info)
        synthetic = transformer.inverse_transform(synthetic, feature_info)

        # real = pd.DataFrame(np.where(mask, real, np.nan), columns=real.columns)
        # synthetic = pd.DataFrame(np.where(mask, synthetic, np.nan), columns=synthetic.columns)

    return real, synthetic
