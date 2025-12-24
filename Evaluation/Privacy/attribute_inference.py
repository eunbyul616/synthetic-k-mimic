import os
import numpy as np
import pandas as pd
from typing import List

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def attribute_inference_attack(real_data: pd.DataFrame,
                               synthetic_data: pd.DataFrame,
                               target_col: str='gender',
                               k: int=5,
                               test_size: float=0.2,
                               random_state: int=42) -> (float, float):
    """
    Evaluates the risk of attribute inference attacks on synthetic data by predicting a sensitive attribute
    (e.g., gender, age, race) using k-Nearest Neighbors (kNN) classifiers trained on real and synthetic data.

    Args:
        real_data: The real dataset containing the sensitive attribute and other features.
        synthetic_data: The synthetic dataset containing the sensitive attribute and other features.
        target_col: The column name of the sensitive attribute to predict. Defaults to 'gender'.
        k: The number of neighbors to use for the kNN classifier. Defaults to 5.
        test_size: The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state: The seed used by the random number generator for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - float: ROC AUC score of the kNN model trained on real data.
            - float: ROC AUC score of the kNN model trained on synthetic data.

     Notes:
        - This function compares the predictability of a sensitive attribute using real and synthetic data.
        - Higher ROC AUC scores indicate a higher risk of attribute inference, as the sensitive attribute
          becomes more predictable.
    """
    def _calculate_auc(y_true: np.array, y_pred: np.array, multi_class: bool=False) -> float:
        """
        Calculates the ROC AUC score for binary or multi-class classification.

        Args:
            y_true: True labels for the data.
            y_pred: Predicted probabilities for each class.
            multi_class: Whether the classification is multi-class. Defaults to False.

        Returns:
            The mean ROC AUC score across classes (for multi-class) or the binary ROC AUC score.
        """

        if multi_class:
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            aucs = []
            for i in range(y_true_bin.shape[1]):
                if len(np.unique(y_true_bin[:, i])) > 1:
                    auc = roc_auc_score(y_true_bin[:, i], y_pred[:, i])
                    aucs.append(auc)
            return np.mean(aucs)
        else:
            return roc_auc_score(y_true, y_pred[:, 1])

    y_real = real_data[target_col]
    X_real = real_data.drop(columns=[target_col])
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(X_real,
                                                                            y_real,
                                                                            test_size=test_size,
                                                                            random_state=random_state)
    y_real_train = np.array(y_real_train, dtype=np.int64)
    y_real_test = np.array(y_real_test, dtype=np.int64)

    nn_model_real = KNeighborsClassifier(n_neighbors=k)
    nn_model_real.fit(X_real_train, y_real_train)
    y_pred_proba_real = nn_model_real.predict_proba(X_real_test)

    y_synthetic = synthetic_data[target_col]
    X_synthetic = synthetic_data.drop(columns=[target_col])
    X_synthetic_train, X_synthetic_test, y_synthetic_train, y_synthetic_test = train_test_split(X_synthetic,
                                                                                                y_synthetic,
                                                                                                test_size=test_size,
                                                                                                random_state=random_state)
    y_synthetic_train = np.array(y_synthetic_train, dtype=np.int64)
    y_synthetic_test = np.array(y_synthetic_test, dtype=np.int64)

    nn_model_synthetic = KNeighborsClassifier(n_neighbors=k)
    nn_model_synthetic.fit(X_synthetic_train, y_synthetic_train)
    y_pred_proba_synthetic = nn_model_synthetic.predict_proba(X_synthetic_test)

    roc_auc_real = _calculate_auc(y_real_test, y_pred_proba_real,
                                  multi_class=(len(np.unique(y_real_train)) > 2))
    roc_auc_synthetic = _calculate_auc(y_synthetic_test, y_pred_proba_synthetic,
                                       multi_class=(len(np.unique(y_synthetic_train)) > 2))

    return roc_auc_real, roc_auc_synthetic

