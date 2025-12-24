import os
import os.path
from types import SimpleNamespace
import yaml
from typing import Any, List
from copy import deepcopy
from omegaconf import OmegaConf


def _load_yaml(cfg_path):
    with open(cfg_path, 'r') as f:
        return _dict_to_namespace(yaml.safe_load(f))


def _dict_to_namespace(d):
    return SimpleNamespace(**{k: _dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def _namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in ns.__dict__.items()}
    elif isinstance(ns, list):
        return [_namespace_to_dict(v) for v in ns]
    else:
        return ns


def compare_namespaces_recursive(ns1: SimpleNamespace, ns2: SimpleNamespace):
    def _compare_recursive(dict1, dict2):
        differences = {}

        for key in dict1:
            if key not in dict2:
                differences[key] = (dict1[key], 'missing in ns2')
            elif isinstance(dict1[key], SimpleNamespace) and isinstance(dict2[key], SimpleNamespace):
                nested_diff = _compare_recursive(dict1[key].__dict__, dict2[key].__dict__)
                if nested_diff:
                    differences[key] = nested_diff
            elif dict1[key] != dict2[key]:
                differences[key] = (dict1[key], dict2[key])

        for key in dict2:
            if key not in dict1:
                differences[key] = ('missing in ns1', dict2[key])
        return differences

    diff = _compare_recursive(ns1.__dict__, ns2.__dict__)
    if 'ignore' in diff.keys():
        diff.pop('ignore')
    return diff


def save_config(cfg: SimpleNamespace):
    # filepath = cfg.path.base_config_file_path.replace('save_name', cfg.log.time)
    save_path = cfg.path.base_config_file_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving config to {save_path}")
    with open(save_path, 'w') as f:
        OmegaConf.save(config=_namespace_to_dict(cfg), f=f)


def set_cfg(cfg: SimpleNamespace, key: str, value: Any):
    keys = key.split('.')

    for k in keys[:-1]:
        cfg = getattr(cfg, k)

    setattr(cfg, keys[-1], value)


def update_cfg(cfg: SimpleNamespace,
               key_list: List[str],
               value_list: List[Any],
               save: bool = False):
    cfg = deepcopy(cfg)
    for key, value in zip(key_list, value_list):
        set_cfg(cfg, key, value)
    if save:
        save_config(cfg)
    else:
        return cfg
