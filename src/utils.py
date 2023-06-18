import os
import argparse
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor


def create_directory(folder_name: str) -> None:
    """
    This method creates a directory with desired name.
    :param folder_name: Name of the directory which is created.
    :return: None.
    """
    folder_path = './' + folder_name
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def fetch_dataset(dataset_name: str, subject_ids: List) -> MOABBDataset:
    """
    This method fetched the required dataset from braindecode.
    :param dataset_name: The name of the dataset which is to be fetched from braindecode MOABBDataset.
    :param subject_ids: The subjects whose data is fetched. This should be a list of the ids i.e. [1, 3, 6].
    :return: The fetched MOABBDataset.
    """
    return MOABBDataset(dataset_name=dataset_name, subject_ids=subject_ids)


def store_dataset(folder_name: str, dataset: MOABBDataset) -> None:
    """
    This method is used to store the fetched dataset in a specific folder.
    :param folder_name: The name of the folder where the dataset needs to be stored.
    :param dataset: The dataset which needs to be stored.
    :return: None.
    """
    dataset.save(path='./' + folder_name)

