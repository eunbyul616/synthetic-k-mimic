import pandas as pd
import numpy as np
import h5py


def stratified_shuffle_split(labels: np.array or pd.Series,
                             patient_ids: np.array or pd.Series=None,
                             train_ratio: float=0.8,
                             test_ratio: float=0.2,
                             val_ratio: float=None):
    unique_labels = np.unique(labels)
    unique_patient_ids = np.unique(patient_ids)
    patient_to_label = {pid: labels[patient_ids == pid].values[0] for pid in unique_patient_ids}

    train_patients, val_patients, test_patients = [], [], []

    for label in unique_labels:
        patients_with_label = [pid for pid in unique_patient_ids if patient_to_label[pid] == label]
        np.random.shuffle(patients_with_label)

        n_train = int(train_ratio * len(patients_with_label))

        if val_ratio is None:
            n_val = int(0.2 * n_train)
        else:
            n_val = int(val_ratio * len(patients_with_label))

        n_test = int(test_ratio * len(patients_with_label))

        train_patients.extend(patients_with_label[:n_train - n_val])
        val_patients.extend(patients_with_label[n_train - n_val:n_train])
        test_patients.extend(patients_with_label[n_train: n_train + n_test])

    train_indices = np.where(np.isin(patient_ids, train_patients))[0]
    val_indices = np.where(np.isin(patient_ids, val_patients))[0]
    test_indices = np.where(np.isin(patient_ids, test_patients))[0]

    return train_indices, val_indices, test_indices


def load_data(data_path: str, hdf_key: str, mode: str='train'):
    data = pd.read_hdf(data_path, key=f'{hdf_key}_{mode}')
    feature_type = load_feature_type(data_path, hdf_key, mode)
    feature_output_dimensions = load_feature_output_dimensions(data_path, hdf_key, mode)

    return data, feature_type, feature_output_dimensions


def load_feature_type(data_path: str, hdf_key: str, mode: str='train'):
    feature_type = dict()
    with h5py.File(data_path, 'r') as f:
        dataset = f[f'{hdf_key}_{mode}']

        for attr_name, attr_value in dataset.attrs.items():
            if attr_name.startswith('feature_type_'):
                feature_type[attr_name.replace('feature_type_', '')] = attr_value

    return feature_type


def load_column_dtypes(data_path: str, hdf_key: str, mode: str='train'):
    column_dtype = dict()
    with h5py.File(data_path, 'r') as f:
        dataset = f[f'{hdf_key}_{mode}']

        for attr_name, attr_value in dataset.attrs.items():
            if attr_name.startswith('column_dtype_'):
                column_dtype[attr_name.replace('column_dtype_', '')] = attr_value

    return column_dtype


def load_feature_output_dimensions(data_path: str, hdf_key: str, mode: str='train'):
    feature_output_dimensions = dict()
    with h5py.File(data_path, 'r') as f:
        dataset = f[f'{hdf_key}_{mode}']

        for attr_name, attr_value in dataset.attrs.items():
            if attr_name.startswith('feature_output_dimensions_'):
                feature_output_dimensions[attr_name.replace('feature_output_dimensions_', '')] = attr_value

    return feature_output_dimensions


def explore_data_structure(name, obj):
    if hasattr(obj, "attrs"):
        print(f"Path: {name}")

        for attr_name, attr_value in obj.attrs.items():
            if attr_name.startswith('dtype_'):
                print(f"  - Attribute: {attr_name} = {attr_value}")
