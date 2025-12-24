import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_holdout_split(data, split_ratio=0.5, random_seed=None):
    from sklearn.model_selection import train_test_split

    train_indices, holdout_indices = train_test_split(
        np.arange(data.shape[0]),
        test_size=(1 - split_ratio),
        random_state=random_seed
    )

    return train_indices, holdout_indices


def _aggregate_once(cfg,
                    static_data, temporal_data,
                    sn_cols, sc_cols, tn_cols, tc_cols,
                    target_col, seq_len, n_timestep,
                    exclude_cols,
                    temporal_agg=True):
    diagnoses_prefix = cfg.preprocess.icd_code.diagnoses_prefix
    procedure_prefix = cfg.preprocess.icd_code.procedure_prefix
    proc_prefix = cfg.preprocess.proc_prefix

    icd_d_cols = [c for c in static_data.columns if c.startswith(diagnoses_prefix) and 'mask' not in c]
    icd_p_cols = [c for c in static_data.columns if c.startswith(procedure_prefix) and 'mask' not in c]
    proc_cols = [c for c in temporal_data.columns if c.startswith(proc_prefix) and 'mask' not in c]

    n = len(static_data)
    static_data = static_data.copy()
    temporal_data = temporal_data.copy()

    static_data['patient_id'] = np.arange(n)
    temporal_data['patient_id'] = np.repeat(static_data['patient_id'].values, seq_len)
    temporal_cols = list(temporal_data.columns)
    new_temporal_data = temporal_data.values.reshape(n, seq_len, -1)[:, :n_timestep, :]
    temporal_data = pd.DataFrame(new_temporal_data.reshape(-1, new_temporal_data.shape[-1]),
                                 columns=temporal_cols)

    for col in sn_cols + sc_cols:
        s_mask = static_data[f'{col}_mask']
        static_data[col] = static_data[col].where(s_mask == 1, np.nan)
    for col in tn_cols + tc_cols:
        t_mask = temporal_data[f'{col}_mask']
        temporal_data[col] = temporal_data[col].where(t_mask == 1, np.nan)

    data = pd.merge(static_data, temporal_data, on='patient_id')
    data = data.drop(columns=['patient_id'])
    mask_cols = [c for c in data.columns if 'mask' in c]
    data = data.drop(columns=mask_cols)

    if temporal_agg:
        # 수치형 집계: 평균
        data[sn_cols + tn_cols] = data[sn_cols + tn_cols].astype(float)
        num_vals = data[sn_cols + tn_cols].values.reshape(-1, n_timestep, len(sn_cols + tn_cols))
        num_data = np.empty((num_vals.shape[0], num_vals.shape[2])); num_data[:] = np.nan
        for i in range(num_vals.shape[2]):
            with np.errstate(all='ignore'):
                num_data[:, i] = np.nanmean(num_vals[:, :, i], axis=1)
        num_data = pd.DataFrame(num_data, columns=sn_cols + tn_cols)

        # 범주형 집계: 최빈값
        cat_vals = data[sc_cols + tc_cols].values.reshape(-1, n_timestep, len(sc_cols + tc_cols))
        cat_data = []
        for i in range(cat_vals.shape[0]):
            row_modes = []
            for j in range(cat_vals.shape[2]):
                s = pd.Series(cat_vals[i, :, j])
                m = s.mode().iloc[0] if not s.mode().empty else np.nan
                row_modes.append(m)
            cat_data.append(row_modes)
        cat_data = pd.DataFrame(cat_data, columns=sc_cols + tc_cols)

        # ICD/PROC: 존재 여부(max)
        if len(icd_d_cols) > 0:
            icd_d_data = np.nanmax(data[icd_d_cols].values.reshape(-1, n_timestep, len(icd_d_cols)), axis=1)
            icd_d_data = pd.DataFrame(icd_d_data, columns=icd_d_cols)
        else:
            icd_d_data = pd.DataFrame(index=num_data.index)

        if len(icd_p_cols) > 0:
            icd_p_data = np.nanmax(data[icd_p_cols].values.reshape(-1, n_timestep, len(icd_p_cols)), axis=1)
            icd_p_data = pd.DataFrame(icd_p_data, columns=icd_p_cols)

        else:
            icd_p_data = pd.DataFrame(index=num_data.index)
        if len(proc_cols) > 0:
            proc_data = np.nanmax(data[proc_cols].values.reshape(-1, n_timestep, len(proc_cols)),  axis=1)
            proc_data = pd.DataFrame(proc_data,  columns=proc_cols)

        else:
            proc_data = pd.DataFrame(index=num_data.index)

        # 타깃: max
        target = np.nanmax(data[target_col].values.reshape(-1, n_timestep), axis=1)
        target = pd.DataFrame(target, columns=[target_col])

        data = pd.concat([num_data, cat_data, icd_d_data, icd_p_data, proc_data], axis=1)
        if target_col in data.columns:
            data = data.drop(columns=[target_col])

        data = pd.concat([data, target], axis=1)

    # 제외 컬럼 처리
    if exclude_cols is not None:
        data = data.drop(columns=[c for c in exclude_cols if c in data.columns])
        categorical_cols = [col for col in sc_cols + tc_cols if (exclude_cols is None or col not in exclude_cols)]
    else:
        categorical_cols = sc_cols + tc_cols

    return data, (sn_cols, tn_cols, categorical_cols)


def preprocess_data(cfg,
                    static_data, temporal_data,
                    sn_cols, sc_cols, tn_cols, tc_cols,
                    target_col, seq_len, n_timestep,
                    exclude_cols=['discharge_location'],
                    fitted=None,
                    drop_first=False,
                    normalize=True,
                    encode=True,
                    temporal_agg=True):
    data, (sn_cols, tn_cols, categorical_cols) = _aggregate_once(
        cfg, static_data, temporal_data,
        sn_cols, sc_cols, tn_cols, tc_cols,
        target_col, seq_len, n_timestep,
        exclude_cols, temporal_agg=temporal_agg
    )

    diagnoses_prefix = cfg.preprocess.icd_code.diagnoses_prefix
    procedure_prefix = cfg.preprocess.icd_code.procedure_prefix
    proc_prefix = cfg.preprocess.proc_prefix

    icd_d_cols = [c for c in static_data.columns if c.startswith(diagnoses_prefix) and 'mask' not in c]
    icd_p_cols = [c for c in static_data.columns if c.startswith(procedure_prefix) and 'mask' not in c]
    proc_cols = [c for c in temporal_data.columns if c.startswith(proc_prefix) and 'mask' not in c]

    if fitted is None:
        fitted = {
            "num_fill": {},     # 각 수치형 컬럼의 결측 대치값(평균)
            "cat_fill": {},     # 각 범주형 컬럼의 결측 대치값(최빈)
            "scalers": {},      # StandardScaler 객체들
            "cat_levels": {},   # 각 범주형 컬럼의 카테고리 레벨 리스트(기준)
            "dummy_columns": None,  # get_dummies 결과 컬럼 스키마
            "drop_first": drop_first,
            "categorical_cols": categorical_cols,
            "sn_cols": sn_cols,
            "tn_cols": tn_cols
        }

        for col in sn_cols + tn_cols:
            fitted["num_fill"][col] = data[col].mean()
        for col in categorical_cols:
            m = data[col].mode()
            fitted["cat_fill"][col] = m.iloc[0] if not m.empty else np.nan

        for col in sn_cols + tn_cols:
            data[col] = data[col].fillna(fitted["num_fill"][col])
        for col in categorical_cols:
            data[col] = data[col].fillna(fitted["cat_fill"][col])

        data[icd_d_cols] = data[icd_d_cols].ffill()
        data[icd_p_cols] = data[icd_p_cols].ffill()
        data[proc_cols] = data[proc_cols].ffill()

        data[icd_d_cols] = data[icd_d_cols].fillna(0)
        data[icd_p_cols] = data[icd_p_cols].fillna(0)
        data[proc_cols] = data[proc_cols].fillna(0)

        for col in categorical_cols:
            levels = pd.Index(pd.Series(data[col].astype("object")).dropna().unique()).tolist()
            if len(levels) == 0:
                levels = [f"__missing__"]
                data[col] = data[col].fillna(levels[0])
            fitted["cat_levels"][col] = levels

        if encode:
            cat_df = {}
            for col in categorical_cols:
                levels = fitted["cat_levels"][col]
                ser = data[col].astype("object")
                ser = ser.where(ser.isin(levels), np.nan).fillna(fitted["cat_fill"][col])
                ser = pd.Categorical(ser, categories=levels)
                cat_df[col] = ser
            cat_df = pd.DataFrame(cat_df)

            encoded = pd.get_dummies(cat_df, drop_first=drop_first)
            encoded = encoded.astype(int)

            data_final = pd.concat([data.drop(columns=categorical_cols), encoded], axis=1)
            fitted["dummy_columns"] = list(encoded.columns)
        else:
            data_final = data

        if normalize:
            for col in sn_cols + tn_cols:
                scaler = StandardScaler()
                data_final[col] = scaler.fit_transform(data_final[[col]])
                fitted["scalers"][col] = scaler
        else:
            for col in sn_cols + tn_cols:
                fitted["scalers"][col] = None

        if target_col not in data_final.columns:
            data_final[target_col] = data[target_col]

        return data_final, fitted

    else:
        for col in fitted["sn_cols"] + fitted["tn_cols"]:
            fillv = fitted["num_fill"].get(col, data[col].mean())
            data[col] = data[col].fillna(fillv)
        for col in fitted["categorical_cols"]:
            fillv = fitted["cat_fill"].get(col, np.nan)
            data[col] = data[col].fillna(fillv)

        data[icd_d_cols] = data[icd_d_cols].ffill()
        data[icd_p_cols] = data[icd_p_cols].ffill()
        data[proc_cols] = data[proc_cols].ffill()

        data[icd_d_cols] = data[icd_d_cols].fillna(0)
        data[icd_p_cols] = data[icd_p_cols].fillna(0)
        data[proc_cols] = data[proc_cols].fillna(0)

        if encode:
            cat_df = {}
            for col in fitted["categorical_cols"]:
                levels = fitted["cat_levels"].get(col, None)
                ser = data[col].astype("object")
                if levels is None or len(levels) == 0:
                    ser = ser.fillna(fitted["cat_fill"].get(col, "__missing__"))
                    levels = [f"__missing__"]
                ser = ser.where(ser.isin(levels), np.nan).fillna(fitted["cat_fill"].get(col, levels[0]))
                cat_df[col] = pd.Categorical(ser, categories=levels)

            cat_df = pd.DataFrame(cat_df)
            encoded = pd.get_dummies(cat_df, drop_first=fitted.get("drop_first", False)).astype(int)
            dummy_cols = fitted["dummy_columns"] or []
            for missing in set(dummy_cols) - set(encoded.columns):
                encoded[missing] = 0
            encoded = encoded.reindex(columns=dummy_cols, fill_value=0)

            data_final = pd.concat([data.drop(columns=fitted["categorical_cols"]), encoded], axis=1)

        else:
            data_final = data

        if normalize:
            for col in fitted["sn_cols"] + fitted["tn_cols"]:
                scaler = fitted["scalers"].get(col, None)
                if scaler is not None:
                    data_final[col] = scaler.transform(data_final[[col]])
                else:
                    pass

        if target_col not in data_final.columns:
            data_final[target_col] = data[target_col]

        return data_final


def select_features(data, exclude_cols=None, include_cols=None):
    if exclude_cols is None and include_cols is None:
        return data
    elif exclude_cols is not None:
        return data.drop(columns=exclude_cols)
    elif include_cols is not None:
        return data[include_cols]
    else:
        raise ValueError("Specify either exclude_cols or include_cols, not both.")