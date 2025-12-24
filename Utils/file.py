import os
import pickle


def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_all_files(path, ext=None, include=None, exclude=None):
    """
    Get all files in the directory
    :param path: str, directory path
    :param ext: str, file extension
    :param include: str, file name to include
    :param exclude: str, file name to exclude
    :return: list, file list
    """
    if ext is not None:
        return [
            x
            for x in os.listdir(path)
            if x.endswith(ext) and (exclude is None or exclude not in x)
        ]
    else:
        return [x for x in os.listdir(path) if exclude is None or exclude not in x]


def get_all_files_recursive(path, ext=None, exclude=None):
    """
    Get all files in the directory recursively
    :param path: str, directory path
    :param ext: str, file extension
    :param exclude: str, file name to exclude
    :return: list, file list
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ext is not None:
                if file.endswith(ext) and (exclude is None or exclude not in file):
                    # if file.endswith(ext) and (exclude is None or file.split('.')[0] not in exclude):
                    file_list.append(os.path.join(root, file))
            else:
                if exclude is None or exclude not in file:
                    file_list.append(os.path.join(root, file))
    return file_list
