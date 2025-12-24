import numpy as np
import pandas as pd
from collections import OrderedDict
from bisect import bisect_left
from scipy.special import softmax


class StochasticNormalization:
    def __init__(self, temp=0.5, decimals=2, rng=None):
        self.info = OrderedDict()
        self.temp = temp
        self.decimals = decimals
        self.rng = np.random.default_rng() if rng is None else rng

    def fit(self, x: pd.DataFrame):
        col = x.columns[0]
        self.params = dict()
        self.info.clear()

        series = x[col].round(self.decimals).dropna()
        vc = series.value_counts().sort_index()
        freq = (vc / vc.sum()).to_numpy()                          # (K,)
        keys = vc.index.to_numpy(dtype=float)                      # (K,)

        scaled = np.exp(np.log1p(freq) / self.temp)
        scaled = scaled / scaled.sum()

        lower = np.concatenate([[0.0], np.cumsum(scaled)[:-1]])    # (K,)
        upper = np.cumsum(scaled)                                  # (K,)

        self.keys = keys
        self.lower = lower
        self.upper = upper
        self.bounds = upper
        self.values = np.asarray(keys)

        for k, lo, up in zip(keys, lower, upper):
            self.params[float(k)] = [float(lo), float(up)]

        self.info[col] = self.params
        self.N = len(series)

    def transform(self, x: pd.DataFrame):
        col = x.columns[0]
        xi = x[col].to_numpy(dtype=float)
        xi = np.round(xi, self.decimals)

        out = np.empty_like(xi, dtype=float)
        nan_mask = np.isnan(xi)
        out[nan_mask] = np.nan
        if (~nan_mask).sum() == 0:
            return out

        z = xi[~nan_mask]

        pos = np.searchsorted(self.keys, z, side='left')
        valid_pos = pos < len(self.keys)
        exact_mask = np.zeros_like(valid_pos, dtype=bool)
        if valid_pos.any():
            exact_mask[valid_pos] = (self.keys[pos[valid_pos]] == z[valid_pos])

        out_nonan = np.empty_like(z, dtype=float)
        if exact_mask.any():
            idx_exact = pos[exact_mask]
            lo = self.lower[idx_exact]
            up = self.upper[idx_exact]
            out_nonan[exact_mask] = self.rng.uniform(lo, up)

        unk = ~exact_mask
        if unk.any():
            pos_unk = pos[unk]
            left = np.clip(pos_unk - 1, 0, len(self.keys) - 1)
            right = np.clip(pos_unk, 0, len(self.keys) - 1)

            zl = self.keys[left]
            zr = self.keys[right]
            zv = z[unk]

            choose_right = np.abs(zv - zr) < np.abs(zv - zl)
            nn = np.where(choose_right, right, left)

            lo = self.lower[nn]
            up = self.upper[nn]
            out_nonan[unk] = self.rng.uniform(lo, up)

        out[~nan_mask] = out_nonan
        return out

    def inverse_transform(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values.ravel()
        x = np.asarray(x, dtype=float)

        out = np.empty_like(x, dtype=float)
        nan_mask = np.isnan(x)
        out[nan_mask] = np.nan
        if (~nan_mask).sum() == 0:
            return out

        v = x[~nan_mask]
        idx = np.searchsorted(self.bounds, v, side='left')
        valid = (idx >= 0) & (idx < len(self.values))
        y = np.full_like(v, np.nan, dtype=float)

        if isinstance(self.values, list):
            self.values = np.asarray(self.values)

        y[valid] = self.values[idx[valid]]
        out[~nan_mask] = y
        return out

    def get_output_sdtypes(self):
        return self.info.keys()


class MinMaxNormalization:
    def __init__(self,
                 feature_range=(0, 1)):
        self.info = OrderedDict()
        self.range = feature_range

    def fit(self, x, min_max_values=None):
        col_name = x.columns[0]

        if min_max_values is not None:
            self.info[col_name] = min_max_values[col_name]
        else:
            self.info[col_name] = {'min': x.min().values[0], 'max': x.max().values[0]}

    def transform(self, x):
        col_name = x.columns[0]
        a, b = self.range[0], self.range[1]
        _x = x[col_name]

        _min, _max = self.info[col_name]['min'], self.info[col_name]['max']
        x_norm = ((_x - _min) / (_max - _min)) * (b - a) + a

        return x_norm.values

    def inverse_transform(self, x):
        col_name = x.columns[0][0]
        a, b = self.range[0], self.range[1]
        _min, _max = self.info[col_name]['min'], self.info[col_name]['max']
        x = x.map(lambda x: x if np.isnan(x) or (x >= a) else a)
        x = x.map(lambda x: x if np.isnan(x) or (x <= b) else b)
        out = ((x - a) / (b - a) * (_max - _min) + _min)
        return out

    def get_output_sdtypes(self):
        return list(self.info.keys())


class Standardization:
    def __init__(self):
        self.info = OrderedDict()

    def fit(self, x):
        mean = np.mean(x.values)
        std = np.std(x.values)
        col_name = x.columns[0]
        self.info[col_name] = [mean, std]

    def transform(self, x):
        col_name = x.columns[0]
        mean, std = self.info[col_name]
        x_norm = (x - mean) / std
        return x_norm.values.flatten()

    def inverse_transform(self, x):
        col_name = x.columns[0][0]
        mean, std = self.info[col_name]
        out = (x * std) + mean
        return out

    def get_output_sdtypes(self):
        return list(self.info.keys())


if __name__ == "__main__":
    x = pd.DataFrame([1, 2, 2, 2, 2, 2, 2, 11, 3, 3, np.nan, None], columns=['dbp'])
    sn = StochasticNormalization()
    sn.fit(x=x)
    x_norm = sn.transform(x=x)
    print(x_norm)

    x_norm = pd.DataFrame(x_norm)
    x = sn.inverse_transform(x_norm)
    print(x)

    x = pd.DataFrame([1, 2, 2, 2, 2, 2, 2, 11, 3, 14, np.nan, None], columns=['dbp'])
    x_norm = sn.transform(x=x)
    print(x_norm)

    x_norm = pd.DataFrame(x_norm)
    x = sn.inverse_transform(x_norm)
    print(x)
    breakpoint()
