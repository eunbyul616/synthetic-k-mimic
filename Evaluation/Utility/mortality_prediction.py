import os
import numpy as np
import pandas as pd
from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from sklearn.neural_network import MLPClassifier


def train_eval_model(model,
                     X_train: pd.DataFrame or np.array,
                     y_train: pd.DataFrame or np.array,
                     X_test: pd.DataFrame or np.array,
                     y_test: pd.DataFrame or np.array,
                     model_type: str='sklearn') -> dict:
    """
        Train and evaluate model.

        Args:
            model: model to train (e.g., LogisticRegression, RandomForest).
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            model_type: Specifies the type of model. Currently supports 'sklearn' models.

        Returns:
            dict: A dictionary containing evaluation metrics:
                  - accuracy: Accuracy score.
                  - auc: Area Under the ROC Curve (None if the model does not support `predict_proba`).
                  - ap: Average Precision score (None if the model does not support `predict_proba`).

        Raises:
            ValueError: If the `model_type` is unsupported.
        """
    if model_type == 'sklearn':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    else:
        raise ValueError('Unknown model type')

    # metrics
    auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
    ap = average_precision_score(y_test, y_pred_prob) if y_pred_prob is not None else None

    return {
        'auc': f'{auc:.3f}' if auc is not None else 'N/A',
        'ap': f'{ap:.3f}' if ap is not None else 'N/A',
        'model': model,
        'y_prob': y_pred_prob,
        'y_true': y_test,

    }


def evaluate_mortality_prediction(train_data: pd.DataFrame,
                                  real_data: pd.DataFrame,
                                  target_col: str,
                                  models: dict=None,
                                  model_type: str='sklearn') -> dict:
    """
    Evaluate mortality prediction performance using real and synthetic datasets.

    Args:
        train_data: Dataset used for training the model, including the target column.
        real_data: Real test dataset for evaluation.
        synthetic_data: Synthetic test dataset for evaluation.
        target_col: Name of the target column representing mortality.
        models: Dictionary of models to use for evaluation. Defaults to logistic regression, GBDT, and random forest.
        model_type: Specifies the type of models being used (currently supports 'sklearn').

    Returns:
       dict: A dictionary containing evaluation results for each model and dataset combination:
              - 'Test on Real': Results for real test data.
              - 'Test on Synthetic': Results for synthetic test data.
    """
    if models is None:
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'GBDT': GradientBoostingClassifier(n_estimators=100),
            'Random Forest':  RandomForestClassifier(n_estimators=100)
        }

    y_train = train_data[target_col].astype(int)
    X_train = train_data.drop(columns=[target_col])
    y_real = real_data[target_col].astype(int)
    X_real = real_data.drop(columns=[target_col])

    res = {}
    for model_name, model in models.items():
        res[model_name] = train_eval_model(model,
                                           X_train,
                                           y_train,
                                           X_real,
                                           y_real,
                                           model_type=model_type)

    return res