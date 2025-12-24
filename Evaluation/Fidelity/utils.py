import numpy as np
import pandas as pd
from Visualization.timeseries import *


def lagged_corr(real, synth, lag=1):
    """
    lagged corr for each feature: corr(X_t, X_{t-lag})
    real, synth: (n_samples, seq_len, n_features)
    """
    n_samples, seq_len, n_features = real.shape
    # t >= lag 인 부분만
    real_x = real[:, lag:, :]           # (n, seq-lag, f)
    real_x_prev = real[:, :-lag, :]     # (n, seq-lag, f)
    synth_x = synth[:, lag:, :]
    synth_x_prev = synth[:, :-lag, :]

    real_corr = []
    synth_corr = []
    for f in range(n_features):
        r = np.corrcoef(real_x[:, :, f].ravel(),
                        real_x_prev[:, :, f].ravel())[0, 1]
        s = np.corrcoef(synth_x[:, :, f].ravel(),
                        synth_x_prev[:, :, f].ravel())[0, 1]
        real_corr.append(r)
        synth_corr.append(s)

    real_corr = np.array(real_corr)
    synth_corr = np.array(synth_corr)

    mad = np.mean(np.abs(real_corr - synth_corr))
    return real_corr, synth_corr, mad


def acf_at_lags(series, lags):
    max_lag = max(lags)
    acf_vals = acf(series, nlags=max_lag, fft=True)
    return {lag: acf_vals[lag] for lag in lags}


def compute_acf_table(real, synth, lags=(1, 12, 24, 29), feature_names=None):
    assert real.shape == synth.shape
    n, T, F = real.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    rows = []
    for f in range(F):
        real_flat = real[:, :, f].ravel()
        synth_flat = synth[:, :, f].ravel()

        real_acf = acf_at_lags(real_flat, lags)
        synth_acf = acf_at_lags(synth_flat, lags)

        for lag in lags:
            rows.append({
                "feature": feature_names[f],
                "lag": lag,
                "acf_real": real_acf[lag],
                "acf_synth": synth_acf[lag],
                "abs_diff": abs(real_acf[lag] - synth_acf[lag])
            })

    df = pd.DataFrame(rows)
    return df


def build_cond_group(df,
                     icu_col="icu_expire_flag",
                     hosp_col="hospital_expire_flag"):
    conds = [
        df[icu_col] == 1,
        df[hosp_col] == 1,
    ]
    choices = ["ICU mortality", "Hospital mortality"]
    return np.select(conds, choices, default="Alive")
