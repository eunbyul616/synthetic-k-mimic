import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from Evaluation.Utility.mortality_prediction import *


def stratified_train_test_split(cfg, _data, _data_hat, target_col, random_seed=None):
    data = _data.copy()
    data_hat = _data_hat.copy()

    label_dist = data[target_col].value_counts(normalize=True)
    n_samples = len(data)

    samples_per_class = {
        cls: int(round(ratio * n_samples))
        for cls, ratio in label_dist.items()
    }
    sampled_dfs = []
    for cls, n_cls_samples in samples_per_class.items():
        candidates = data_hat[data_hat[target_col] == cls]
        assert len(candidates) >= n_cls_samples, f"Not enough samples of class {cls} in data_hat"
        sampled_cls = candidates.sample(n=n_cls_samples, random_state=random_seed)
        sampled_dfs.append(sampled_cls)

    sampled_data_hat = shuffle(pd.concat(sampled_dfs), random_state=cfg.seed).reset_index(drop=True)

    return sampled_data_hat


def stratified_train_test_split_incremental(cfg, _data, _data_hat, target_col, rates,
                                            extract_half_flag=False,
                                            random_seed=42):
    data = _data.copy()
    data_hat = _data_hat.copy()

    if extract_half_flag:
        label_dist = {0: 0.5, 1: 0.5}
    else:
        label_dist = data[target_col].value_counts(normalize=True)

    total_samples = len(data)

    if 'patient_id' not in data_hat.columns:
        data['patient_id'] = np.arange(len(data))
        data_hat['patient_id'] = np.arange(len(data_hat))

    sampled_per_rate = {}

    prev_rate = None

    for rate in sorted(rates):
        n_samples = int(total_samples * rate)

        samples_per_class = {
            cls: int(round(ratio * n_samples))
            for cls, ratio in label_dist.items()
        }

        prev_df = sampled_per_rate.get(prev_rate, None)

        sampled_dfs = []
        for cls, n_cls_samples in samples_per_class.items():
            n_prev_cls = 0 if prev_df is None else int((prev_df[target_col] == cls).sum())
            n_to_sample = n_cls_samples - n_prev_cls

            if n_to_sample > 0:
                candidates = data_hat[data_hat[target_col] == cls]
                sampled_cls = candidates.sample(n=n_to_sample, replace=True, random_state=random_seed)
                sampled_dfs.append(sampled_cls)
            else:
                continue

        if sampled_dfs:
            new_samples = pd.concat(sampled_dfs, axis=0)
            if prev_df is not None:
                cur_df = pd.concat([prev_df, new_samples], axis=0)
            else:
                cur_df = new_samples
            sampled_per_rate[rate] = shuffle(cur_df, random_state=random_seed).reset_index(drop=True)
        else:
            sampled_per_rate[rate] = prev_df if prev_df is not None else None

        prev_rate = rate

    return sampled_per_rate



def select_features(data, exclude_cols=None, include_cols=None):
    if exclude_cols is None and include_cols is None:
        return data
    elif exclude_cols is not None:
        return data.drop(columns=exclude_cols)
    elif include_cols is not None:
        return data[include_cols]
    else:
        raise ValueError("Specify either exclude_cols or include_cols, not both.")


def bootstrap_evaluation_stratified(cfg,
                                    _data,
                                    _data_hat,
                                    target_col,
                                    models,
                                    rates=[0.1, 0.25, 0.5],
                                    train_size=0.8,
                                    n_bootstrap=100,
                                    extract_half_flag=False,
                                    exclude_cols=None,
                                    include_cols=None,
                                    seed=42):
    data = _data.copy()
    data_hat = _data_hat.copy()
    model_names = ("LogisticRegression", "RandomForest", "GBDT")

    results_real = {rate: {model: {'auc': [], 'ap': [], 'y_prob': [], 'y_true': []} for model in models.keys()} for rate in rates}
    results_synth = {model: {'auc': [], 'ap': [], 'y_prob': [], 'y_true': []} for model in models.keys()}

    for i in tqdm(range(n_bootstrap)):
        if n_bootstrap > 1:
            seed = np.random.randint(0, 1e3)

        sampled_dfs_dict = stratified_train_test_split_incremental(cfg, data, data_hat, target_col, rates,
                                                                   extract_half_flag=extract_half_flag,
                                                                   random_seed=seed)

        syn = stratified_train_test_split(cfg,
                                          data,
                                          data_hat,
                                          target_col=target_col,
                                          random_seed=seed)

        labels_hat = syn[target_col].values
        synth_train_indices, synth_test_indices = train_test_split(
            np.arange(labels_hat.shape[0]),
            stratify=labels_hat,
            train_size=train_size,
            random_state=seed,
        )
        synthetic_train_data = syn.iloc[synth_train_indices]

        # real
        labels = data[target_col].values
        real_train_indices, real_test_indices = train_test_split(
            np.arange(labels.shape[0]),
            stratify=labels,
            train_size=train_size,
            random_state=seed,
        )
        real_train_data = data.iloc[real_train_indices]
        real_test_data = data.iloc[real_test_indices]

        real_train_data = select_features(real_train_data, exclude_cols=exclude_cols, include_cols=include_cols)
        real_test_data = select_features(real_test_data, exclude_cols=exclude_cols, include_cols=include_cols)
        synthetic_train_data = select_features(synthetic_train_data, exclude_cols=exclude_cols, include_cols=include_cols)

        eval_synth = evaluate_mortality_prediction(synthetic_train_data, real_test_data, target_col, models)
        for model_name in models.keys():
            results_synth[model_name]['auc'].append(float(eval_synth[model_name]['auc']))
            results_synth[model_name]['ap'].append(float(eval_synth[model_name]['ap']))
            results_synth[model_name]['y_prob'].append(eval_synth[model_name]['y_prob'])
            results_synth[model_name]['y_true'].append(eval_synth[model_name]['y_true'])

        for rate in rates:
            print(f'Bootstrap Iteration {i+1}/{n_bootstrap}, Rate: {rate}')
            sampled_synthetic_train_data = sampled_dfs_dict[rate]
            if sampled_synthetic_train_data is not None:
                sampled_synthetic_train_data = sampled_synthetic_train_data.drop(columns=['patient_id'])
                sampled_synthetic_train_data = select_features(sampled_synthetic_train_data,
                                                               exclude_cols=exclude_cols,
                                                               include_cols=include_cols)
                train_data = pd.concat([real_train_data, sampled_synthetic_train_data], axis=0).reset_index(drop=True)
            else:
                train_data = real_train_data

            eval_real = evaluate_mortality_prediction(train_data, real_test_data, target_col, models)
            for model_name in models.keys():
                results_real[rate][model_name]['auc'].append(float(eval_real[model_name]['auc']))
                results_real[rate][model_name]['ap'].append(float(eval_real[model_name]['ap']))
                results_real[rate][model_name]['y_prob'].append(eval_real[model_name]['y_prob'])
                results_real[rate][model_name]['y_true'].append(eval_real[model_name]['y_true'])

    model_order = ["GBDT", "Random Forest", "Logistic Regression"]
    metric_info = {
        "AUC": ("auc", lambda x: f"{np.mean(x):.3f} ± {np.std(x):.3f}"),
        "AP": ("ap", lambda x: f"{np.mean(x):.3f} ± {np.std(x):.3f}")
    }
    records_dict = {}
    for rate in rates:
        records = []
        for model in models.keys():
            for metric_name, (metric_key, formatter) in metric_info.items():
                score = results_real[rate][model][metric_key]
                records.append({
                    "Target": target_col,
                    "Model": model,
                    "Metrics": metric_name,
                    "TRTR": formatter(score)
                })
        df = pd.DataFrame(records)
        df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
        df["Metrics"] = pd.Categorical(df["Metrics"], categories=metric_info.keys(), ordered=True)
        df = df.sort_values(["Model", "Metrics"]).reset_index(drop=True)
        records_dict[rate] = df

    combined_df = pd.concat([df.assign(rate=rate) for rate, df in records_dict.items()], axis=0).reset_index(drop=True)
    combined_df = combined_df.pivot_table(index=['Target', 'Model', 'Metrics'], columns='rate', values='TRTR', aggfunc='first')
    records = []
    for model in models.keys():
        for metric_name, (metric_key, formatter) in metric_info.items():
            score = results_synth[model][metric_key]
            records.append({
                "Target": target_col,
                "Model": model,
                "Metrics": metric_name,
                "TSTR": formatter(score)
            })
        df = pd.DataFrame(records)
        df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
        df["Metrics"] = pd.Categorical(df["Metrics"], categories=metric_info.keys(), ordered=True)
        df = df.sort_values(["Model", "Metrics"]).reset_index(drop=True)
    combined_df = pd.merge(combined_df, df, on=['Target', 'Model', 'Metrics'], how='left')

    return combined_df, results_real, results_synth



def tune_mortality_models(data, target_col, model_name, n_trials=100, random_state=42, train_size=0.8,
                          exclude_cols=None, include_cols=None, seed=42):
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    labels = data[target_col].values
    real_train_indices, real_test_indices = train_test_split(
        np.arange(labels.shape[0]),
        stratify=labels,
        train_size=train_size,
        random_state=seed,
    )
    real_train_data = data.iloc[real_train_indices]
    real_train_data = select_features(real_train_data, exclude_cols=exclude_cols, include_cols=include_cols)

    X = real_train_data.drop(columns=[target_col])
    y = real_train_data[target_col]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    def plateau_stop(study, trial, window=30, min_gain=0.002):
        vals = [t.value for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if len(vals) <= window:
            return
        hist_best = float('inf')
        improved = False
        for i, v in enumerate(vals):
            if v < hist_best - 1e-12:
                hist_best = v
            if i >= len(vals) - window and hist_best - v >= min_gain:
                improved = True
                break
        if not improved:
            study.stop()

    def get_proba(clf, Xv):
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(Xv)[:, 1]
        if hasattr(clf, "decision_function"):
            s = clf.decision_function(Xv)
            return 1 / (1 + np.exp(-s))
        raise ValueError("Cannot get probability estimates for the classifier")

    def objective(trial):
        if model_name == "LogisticRegression":
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
            class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
            max_iter = trial.suggest_int("max_iter", 300, 2000)
            clf = LogisticRegression(C=C, penalty="l2", solver=solver,
                                     class_weight=class_weight, max_iter=max_iter)

        elif model_name == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 100, 800)
            max_depth = trial.suggest_int("max_depth", 2, 32)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
            clf = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    max_features=max_features, class_weight=class_weight,
                    random_state=random_state, n_jobs=-1
                )

        else:
            n_estimators = trial.suggest_int("n_estimators", 100, 800)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            max_depth = trial.suggest_int("max_depth", 1, 5)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            clf = GradientBoostingClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate,
                    max_depth=max_depth, subsample=subsample,
                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        aucs = []
        for tr, va in skf.split(X, y):
            if isinstance(X, np.ndarray):
                X_tr, X_va = X[tr], X[va]
                y_tr, y_va = y[tr], y[va]
            else:
                X_tr, X_va = X.iloc[tr], X.iloc[va]
                y_tr, y_va = y.iloc[tr], y.iloc[va]

            clf.fit(X_tr, y_tr)
            p = get_proba(clf, X_va)
            aucs.append(roc_auc_score(y_va, p))

        return 1.0 - float(np.mean(aucs))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", study_name=model_name, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, callbacks=[lambda s, t: plateau_stop(s,t,30,0.002)])

    return study


def get_model(cfg, best_params):
    models = {
        'GBDT': GradientBoostingClassifier(
            n_estimators=best_params['GBDT']["n_estimators"],
            learning_rate=best_params['GBDT']["learning_rate"],
            max_depth=best_params['GBDT']["max_depth"],
            subsample=best_params['GBDT']["subsample"],
            min_samples_split=best_params['GBDT']["min_samples_split"],
            min_samples_leaf=best_params['GBDT']["min_samples_leaf"],
            random_state=cfg.seed
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=best_params['RandomForest']["n_estimators"],
            max_depth=best_params['RandomForest']["max_depth"],
            min_samples_split=best_params['RandomForest']["min_samples_split"],
            min_samples_leaf=best_params['RandomForest']["min_samples_leaf"],
            max_features=best_params['RandomForest']["max_features"],
            class_weight=best_params['RandomForest']["class_weight"],
            random_state=cfg.seed,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            C=best_params['LogisticRegression']["C"],
            penalty="l2",
            solver=best_params['LogisticRegression']["solver"],
            class_weight=best_params['LogisticRegression']["class_weight"],
            max_iter=best_params['LogisticRegression']["max_iter"]
        ),
    }

    return models


