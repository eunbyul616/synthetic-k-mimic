import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

import config_manager

from Datasets.dataset_k_mimic import KMIMICDataset as CustomDataset

from Evaluation.Utility.utils import *
from Evaluation.Metrics.compute_pvalue import *
from Evaluation.preprocess import *


if __name__ == '__main__':
    config_manager.load_config()
    eval_cfg = config_manager.config

    cols = None
    dataset_name = eval_cfg.dataset.dataset_name
    fname = eval_cfg.preprocess.preprocess_fname_suffix
    train_ratio, test_ratio = eval_cfg.preprocess.train_valid_test_split_ratio
    dataset_fname = f'{dataset_name}_{fname}_{int(train_ratio * 10)}.h5'

    dataset = CustomDataset(cfg=eval_cfg,
                            dataset_name=dataset_name,
                            dataset_fname=dataset_fname,
                            mode='test',
                            condition_col=eval_cfg.data.condition_col,
                            static_cols=None)


    model_name = eval_cfg.evaluation.model_name
    checkpoint = eval_cfg.evaluation.checkpoint
    print(checkpoint)
    eval_file_path = os.path.join(eval_cfg.path.eval_file_path, model_name, checkpoint)
    print(eval_file_path)

    target_col = 'hospital_expire_flag'
    extract_half_flag = False
    feature_mode = 'feature_set'  # 'all' or 'feature_set'

    seq_len = eval_cfg.dataloader.seq_len
    n_timestep = 24

    train_size = train_ratio

    n_bootstrap = 100
    n_trials = 100
    load = False

    static_transformer = dataset.static_transformer
    temporal_transformer = dataset.temporal_transformer

    sc_cols = dataset.sc_cols
    tc_cols = dataset.tc_cols
    sn_cols = dataset.sn_cols
    tn_cols = dataset.tn_cols
    sl_cols = dataset.sl_cols
    tl_cols = dataset.tl_cols

    diagnoses_prefix = eval_cfg.preprocess.icd_code.diagnoses_prefix
    procedure_prefix = eval_cfg.preprocess.icd_code.procedure_prefix
    proc_prefix = eval_cfg.preprocess.proc_prefix

    # read data
    static_data = pd.read_csv(os.path.join(eval_file_path, 'static_data.csv'))
    temporal_data = pd.read_csv(os.path.join(eval_file_path, 'temporal_data.csv'))
    static_data_hat = pd.read_csv(os.path.join(eval_file_path, 'static_reconstructed_26560.csv'))
    temporal_data_hat = pd.read_csv(os.path.join(eval_file_path, 'temporal_reconstructed_26560.csv'))
    save_eval_res_path = os.path.join(eval_file_path, 'evaluation')
    os.makedirs(save_eval_res_path, exist_ok=True)

    # preprocess data
    data, fitted = preprocess_data(eval_cfg, static_data, temporal_data,
                                   sn_cols, sc_cols, tn_cols, tc_cols,
                                   target_col, seq_len, n_timestep,
                                   exclude_cols=['discharge_location'],
                                   fitted=None,  # fit
                                   drop_first=False)
    data_hat = preprocess_data(eval_cfg, static_data_hat, temporal_data_hat,
                               sn_cols, sc_cols, tn_cols, tc_cols,
                               target_col, seq_len, n_timestep,
                               exclude_cols=['discharge_location'],
                               fitted=fitted,
                               drop_first=False)

    icd_d_cols = [col for col in data.columns if (col.startswith(diagnoses_prefix)) and ('mask' not in col)]
    icd_p_cols = [col for col in data.columns if (col.startswith(procedure_prefix)) and ('mask' not in col)]
    proc_cols = [col for col in data.columns if (col.startswith(proc_prefix)) and ('mask' not in col)]

    if target_col == 'icu_expire_flag':
        if feature_mode == 'all':
            exclude_cols = ['discharge_location', 'los'] + icd_d_cols + icd_p_cols

        else:
            include_cols = [
                'age',
                'RR',
                'HR',
                'SBP',
                'DBP',
                'BT',
                'SpO2',
                'Glasgow coma scale(eye)',
                'Glasgow coma scale(verbal)',
                'Glasgow coma scale(motor)',
                'ALT',
                'AST',
                'Albumin',
                'BUN',
                'Bilirubin, total',
                'Chloride, serum',
                'Creatinine',
                'Glucose',
                'Hb',
                'PT (INR)',
                'PLT',
                'Potassium, serum',
                'Sodium, serum',
                'WBC',
                'aPTT',
            ] + [target_col]
            # CRP

    else:
        if feature_mode == 'all':
            exclude_cols = ['discharge_location', 'los'] + icd_d_cols + icd_p_cols
        else:
            include_cols = [
                'age',
                'RR',
                'HR',
                'SBP',
                'DBP',
                'BT',
                'SpO2',
                'Glasgow coma scale(eye)',
                'Glasgow coma scale(verbal)',
                'Glasgow coma scale(motor)',
                'ALT',
                'AST',
                'Albumin',
                'BUN',
                'Bilirubin, total',
                'Chloride, serum',
                'Creatinine',
                'Glucose',
                'Hb',
                'PT (INR)',
                'PLT',
                'Potassium, serum',
                'Sodium, serum',
                'WBC',
                'aPTT',
            ] + [target_col]
            # CRP


    model_names = ("LogisticRegression", "RandomForest", "GBDT")
    if not load or (
            not os.path.exists(os.path.join(save_eval_res_path, f'best_params_{target_col}_kmimic_trtr.pkl'))
    ):
        studies, best_params, best_models, best_scores = {}, {}, {}, {}
        for model_name in model_names:
            study = tune_mortality_models(data, target_col,
                                          model_name=model_name,
                                          exclude_cols=exclude_cols if feature_mode == 'all' else None,
                                          include_cols=include_cols if feature_mode != 'all' else None,
                                          n_trials=n_trials,
                                          train_size=train_size,
                                          random_state=eval_cfg.seed)
            studies[model_name] = study
            bp = study.best_trial.params
            best_params[model_name] = bp
            best_scores[model_name] = 1.0 - study.best_value

        with open(os.path.join(save_eval_res_path, f'best_params_{target_col}_kmimic_trtr.pkl'), 'wb') as f:
            pickle.dump(best_params, f)
    else:
        with open(os.path.join(save_eval_res_path, f'best_params_{target_col}_kmimic_trtr.pkl'), 'rb') as f:
            best_params = pickle.load(f)
    models = get_model(eval_cfg, best_params)

    if not load or (
            not os.path.exists(os.path.join(save_eval_res_path, f'best_params_{target_col}_kmimic_tstr.pkl'))
    ):
        studies, best_params, best_models, best_scores = {}, {}, {}, {}
        for model_name in model_names:
            study = tune_mortality_models(data_hat, target_col,
                                          model_name=model_name,
                                          exclude_cols=exclude_cols if feature_mode == 'all' else None,
                                          include_cols=include_cols if feature_mode != 'all' else None,
                                          n_trials=n_trials,
                                          train_size=train_size,
                                          random_state=eval_cfg.seed)
            studies[model_name] = study
            bp = study.best_trial.params
            best_params[model_name] = bp
            best_scores[model_name] = 1.0 - study.best_value

        with open(os.path.join(save_eval_res_path, f'best_params_{target_col}_kmimic_tstr.pkl'), 'wb') as f:
            pickle.dump(best_params, f)
    else:
        with open(os.path.join(save_eval_res_path, f'best_params_{target_col}_kmimic_tstr.pkl'), 'rb') as f:
            best_params = pickle.load(f)
    models_syn = get_model(eval_cfg, best_params)

    df, res_real_bootstrap, res_syn_bootstrap = bootstrap_evaluation_stratified(eval_cfg,
                                                                                data,
                                                                                data_hat,
                                                                                target_col=target_col,
                                                                                models=models,
                                                                                rates=[0.0, 0.1, 0.25, 0.5],
                                                                                train_size=train_size,
                                                                                n_bootstrap=n_bootstrap,
                                                                                extract_half_flag=extract_half_flag,
                                                                                exclude_cols=exclude_cols if feature_mode == 'all' else None,
                                                                                include_cols=include_cols if feature_mode != 'all' else None)

    _, res_real, res_syn = bootstrap_evaluation_stratified(eval_cfg,
                                              data,
                                              data_hat,
                                              target_col=target_col,
                                              models=models,
                                              rates=[0.0, 0.1, 0.25, 0.5],
                                              train_size=train_size,
                                              n_bootstrap=1,
                                              extract_half_flag=extract_half_flag,
                                              exclude_cols=exclude_cols if feature_mode == 'all' else None,
                                              include_cols=include_cols if feature_mode != 'all' else None)

    p_value_dict = {model_name: {
        'TSTR': {},
        0.1: {},
        0.25: {},
        0.5: {}
    } for model_name in res_syn_bootstrap.keys()}
    for model_name in res_syn_bootstrap.keys():
        for rate in [0.1, 0.25, 0.5]:
            incre_p_value = compute_auprc_p_value(res_real_bootstrap[0.0][model_name]['ap'],
                                                  res_real_bootstrap[rate][model_name]['ap'])
            star, _ = p_value_to_star(incre_p_value)
            p_value_dict[model_name][rate]['ap'] = f'({star})'

            incre_auc_p_value = compute_auc_p_value(res_real[0.0][model_name]['y_true'][0],
                                                    res_real[0.0][model_name]['y_prob'][0],
                                                    res_real[rate][model_name]['y_prob'][0])
            star, _ = p_value_to_star(incre_auc_p_value)
            p_value_dict[model_name][rate]['auc'] = f'({star})'  # f'{np.round(kmimic_incre_auc_p_value, 3)} ({star})'

        syn_kmimic_p_value = compute_auprc_p_value(res_real_bootstrap[0.0][model_name]['ap'],
                                                   res_syn_bootstrap[model_name]['ap'])
        star, _ = p_value_to_star(syn_kmimic_p_value)
        p_value_dict[model_name]['TSTR']['ap'] = f'({star})'

        syn_kmimic_auc_p_value = compute_auc_p_value(res_real[0.0][model_name]['y_true'][0],
                                                     res_real[0.0][model_name]['y_prob'][0],
                                                     res_syn[model_name]['y_prob'][0])
        star, _ = p_value_to_star(syn_kmimic_auc_p_value)
        p_value_dict[model_name]['TSTR']['auc'] = f'({star})'

    p_value_df = pd.concat({k: pd.DataFrame(v).T for k, v in p_value_dict.items()})
    p_value_df.index.names = ['Model', 'Dataset']
    p_value_df.reset_index(inplace=True)
    long = p_value_df.melt(id_vars=['Model', 'Dataset'],
                           value_vars=['ap', 'auc'],
                           var_name='Metrics',
                           value_name='p_value')
    wide = long.pivot(index=['Model', 'Metrics'], columns='Dataset', values='p_value').reset_index()
    wide = wide.rename(columns={c: f"{c} (p)" for c in wide.columns if c not in ['Model', 'Metrics']})
    wide['Metrics'] = wide['Metrics'].str.upper()

    merged_df = pd.merge(df, wide, on=['Model', 'Metrics'], how='left')
    for c in df.columns:
        if c not in ['Target', 'Model', 'Metrics', 0.0]:
            merged_df[c] = merged_df[c] + ' ' + merged_df[f'{c} (p)']
            merged_df = merged_df.drop(columns=[f'{c} (p)'])

    merged_df.to_excel(os.path.join(save_eval_res_path, f'stratified_bootstrap_evaluation_ratio_{target_col}_{feature_mode}_{str(extract_half_flag)}.xlsx'), index=False)

