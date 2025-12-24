import os
import numpy as np
import pandas as pd
from typing import List

from scipy.stats import entropy
from sklearn.metrics import pairwise_distances


def calculate_k_anonymity(data: pd.DataFrame, quasi_identifiers: List[str]) -> int:
    """
    Calculate the k-anonymity for the given dataset based on the quasi-identifiers.
    Args:
        data: dataset to calculate the k-anonymity
        quasi_identifiers: columns used as quasi-identifiers

    Returns:
        k-anonymity value, representing the size of the smallest equivalence class.
    """
    grouped = data.groupby(quasi_identifiers).size()
    k_anonymity = grouped.min()

    return k_anonymity


def calculate_l_diversity(data: pd.DataFrame, quasi_identifiers: List[str], sensitive_attribute: str) -> int:
    """
    Calculate l-diversity for the given dataset based on the quasi-identifiers and sensitive attribute.

    Args:
        data: dataset to calculate the l-diversity.
        quasi_identifiers: columns used as quasi-identifiers.
        sensitive_attribute: sensitive attribute to evaluate diversity.

    Returns:
        l-diversity value, representing the minimum number of distinct sensitive values.
    """
    grouped = data.groupby(quasi_identifiers)[sensitive_attribute].nunique()
    l_diversity = grouped.min()

    return l_diversity


def calculate_t_closeness(data: pd.DataFrame, quasi_identifiers: List[str], sensitive_attribute: str) -> float:
    """
    Calculate t-closeness for the given dataset based on the quasi-identifiers and sensitive attribute.

    Args:
        data: dataset to calculate the t-closeness.
        quasi_identifiers: columns used as quasi-identifiers.
        sensitive_attribute: sensitive attribute to evaluate closeness.

    Returns:
        t-closeness value, representing the maximum divergence from the global distribution.

    """
    # Calculate the global distribution of the sensitive attribute
    total_distribution = data[sensitive_attribute].value_counts(normalize=True)

    max_t_closeness = 0
    for _, group in data.groupby(quasi_identifiers):
        # Calculate the group's distribution
        group_distribution = group[sensitive_attribute].value_counts(normalize=True)
        group_distribution = group_distribution.reindex(total_distribution.index, fill_value=0)

        # compute the divergence using entropy
        divergence = entropy(group_distribution, total_distribution)
        max_t_closeness = max(max_t_closeness, divergence)

    return max_t_closeness



def calculate_single_out_risk(original_data: np.array, synthetic_data: np.array):
    """
    Calculate the identifiability risk of the synthetic data.

    Args:
        original_data (np.ndarray): Original data of shape (num_patients, num_features).
        synthetic_data (np.ndarray): Synthetic data of shape (num_patients, num_features).

    Returns:
        float: The identifiability risk of the synthetic data.
    """
    # Ensure data consistency
    assert original_data.shape == synthetic_data.shape, "Data shapes must match."

    original_set = set(map(tuple, original_data))
    matches = [1 if tuple(row) in original_set else 0 for row in synthetic_data]

    return np.mean(matches)


def calculate_inferential_disclosure_risk(original_data: np.array, synthetic_data: np.array):
    """
    Calculate the inference risk of the synthetic data.

    Args:
        original_data (np.ndarray): Original data of shape (num_patients, num_features).
        synthetic_data (np.ndarray): Synthetic data of shape (num_patients, num_features).

    Returns:
        float: The inference risk of the synthetic data.
    """
    # Ensure data consistency
    assert original_data.shape == synthetic_data.shape, "Data shapes must match."

    dist_synthetic_to_original = pairwise_distances(synthetic_data, original_data)
    dist_original_to_original = pairwise_distances(synthetic_data, original_data)

    np.fill_diagonal(dist_original_to_original, np.inf)

    # Calculate d_s and d_o
    d_s = np.min(dist_synthetic_to_original, axis=1)
    d_o = np.min(dist_original_to_original, axis=1)

    closest_original_indices = np.argmin(dist_synthetic_to_original, axis=1)
    d_o_closest = d_o[closest_original_indices]

    # Compare distances to compute inference risk
    inference_risk = np.mean(d_s < d_o_closest)

    return inference_risk


def calculate_correct_attribution_probability(original_data: np.ndarray,
                                              synthetic_data: np.ndarray,
                                              quasi_identifier_indices: list,
                                              sensitive_index: int or list) -> float:
    """
    Calculate Correct Attribution Probability (CAP) between original and synthetic datasets.
    Measures how often the closest match (by quasi-identifiers) in original data has the same sensitive attribute as the synthetic sample.

    Args:
        original_data (np.ndarray): Original dataset of shape (N, D).
        synthetic_data (np.ndarray): Synthetic dataset of shape (N, D).
        quasi_identifier_indices (list): List of column indices used as quasi-identifiers (K).
        sensitive_index (int): Index of the sensitive attribute column (T).

    Returns:
        float: CAP value (between 0 and 1).
    """
    assert original_data.shape == synthetic_data.shape, "original_data and synthetic_data must have the same shape"

    # Extract quasi-identifiers (K) and sensitive attribute (T)
    K_orig = original_data[:, quasi_identifier_indices]
    K_syn = synthetic_data[:, quasi_identifier_indices]

    T_orig = original_data[:, sensitive_index]
    T_syn = synthetic_data[:, sensitive_index]

    # Compute pairwise distances between synthetic and original using K
    dist = pairwise_distances(K_syn, K_orig)

    # Find nearest original record for each synthetic record
    nearest_indices = np.argmin(dist, axis=1)

    # Compare sensitive attribute
    matched_T = T_orig[nearest_indices]
    cap = np.all(matched_T == T_syn, axis=-1).mean()

    return cap


def calculate_thresholds(data: np.ndarray, iterations: int = 100, quantiles: list = [0.9, 0.95, 0.99], save_path: str = None):
    """
    Calculate thresholds for identifiability and inference risks.

    Parameters:
        data (np.ndarray): The original dataset.
        iterations (int): Number of iterations for random splits.
        quantiles (list): Quantiles for threshold calculation.

    Returns:
        dict: Thresholds for identifiability and inference risks.
    """
    identifiability_risks = []
    inference_risks = []

    for _ in range(iterations):
        # Randomly split data into A and B
        np.random.shuffle(data)
        split_idx = len(data) // 2
        original_A = data[:split_idx]
        synthetic_B = data[split_idx:split_idx*2]

        original_A = np.nan_to_num(original_A, nan=-1)
        synthetic_B = np.nan_to_num(synthetic_B, nan=-1)

        # Calculate risks
        identifiability_risk = calculate_single_out_risk(original_A, synthetic_B)

        # original_A = np.nan_to_num(original_A, nan=-1)
        # synthetic_B = np.nan_to_num(synthetic_B, nan=-1)

        inference_risk = calculate_inferential_disclosure_risk(original_A, synthetic_B)

        # Adjust identifiability risk
        adjusted_identifiability_risk = 1 - (1 - identifiability_risk) ** 2

        identifiability_risks.append(adjusted_identifiability_risk)
        inference_risks.append(inference_risk)

    # Calculate thresholds
    identifiability_thresholds = {f"{int(q * 100)}%": np.quantile(identifiability_risks, q) for q in quantiles}
    inference_thresholds = {f"{int(q * 100)}%": np.quantile(inference_risks, q) for q in quantiles}

    return {
        'single_out_risk_thresholds': {k: np.round(v, 3) for k, v in identifiability_thresholds.items()},
        'inferential_disclosure_risk_thresholds': {k: np.round(v, 3) for k, v in inference_thresholds.items()}
    }


def calculate_cap_thresholds(data: np.ndarray,
                             quasi_identifier_indices: list,
                             sensitive_index: int or list,
                             iterations: int = 100,
                             quantiles: list = [0.9, 0.95, 0.99]):
    """
    Calculate thresholds for identifiability and inference risks.

    Parameters:
        data (np.ndarray): The original dataset.
        iterations (int): Number of iterations for random splits.
        quantiles (list): Quantiles for threshold calculation.

    Returns:
        dict: Thresholds for identifiability and inference risks.
    """
    attribute_disclosure_risks = []

    for _ in range(iterations):
        # Randomly split data into A and B
        np.random.shuffle(data)
        split_idx = len(data) // 2
        original_A = data[:split_idx]
        synthetic_B = data[split_idx:split_idx*2]

        original_A = np.nan_to_num(original_A, nan=-1)
        synthetic_B = np.nan_to_num(synthetic_B, nan=-1)

        cap = calculate_correct_attribution_probability(original_A, synthetic_B, quasi_identifier_indices, sensitive_index)
        adjusted_cap = 1 - (1 - cap) ** 2
        attribute_disclosure_risks.append(adjusted_cap)

    attribute_disclosure_risk_thresholds = {f"{int(q * 100)}%": np.quantile(attribute_disclosure_risks, q) for q in quantiles}

    return {
        'attribute_disclosure_risk_thresholds': {k: np.round(v, 3) for k, v in attribute_disclosure_risk_thresholds.items()}
    }


if __name__ == "__main__":
    data = pd.DataFrame({
        'age': np.random.randint(18, 90, 1000),
        'sex': np.random.choice(['M', 'F'], 1000),
        'mortality': np.random.choice([0, 1], 1000),
        'icd10_pcs': np.random.choice(['A', 'B', 'C'], 1000),
        'spo2': np.random.randint(90, 100, 1000),
        'heart_rate': np.random.randint(60, 100, 1000),
        'temperature': np.random.randint(36, 38, 1000)
    })

    quasi_identifiers = ['age', 'sex']
    sensitive_attribute = 'icd10_pcs'

    k = calculate_k_anonymity(data, quasi_identifiers)
    print(f'K-anonymity: {k}')
    l = calculate_l_diversity(data, quasi_identifiers, sensitive_attribute)
    print(f'L-diversity: {l}')
    t = calculate_t_closeness(data, quasi_identifiers, sensitive_attribute)
    print(f'T-closeness: {t}')