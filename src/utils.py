import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from braindecode.datasets import MOABBDataset
from braindecode.datautil import load_concat_dataset
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


def load_dataset(folder_name: str) -> MOABBDataset:
    """
    This method is used to load the fetched dataset from a specific folder.
    :param folder_name: The name of the folder where the dataset is stored.
    :return: Loads and returns the stored dataset.
    """
    return load_concat_dataset(path='./' + folder_name, preload=True)


def dataset_preprocessor(data: MOABBDataset,
                         l_freq: float = 4.0,
                         h_freq: float = 38.0,
                         ems_factor: float = 1e-3,
                         init_block_size: int = 1000) -> MOABBDataset:
    """
    This method applies all the required preprocessing to the MOABBDataset.
    :param data: The dataset which requires preprocessing.
    :param l_freq: The lower limit of the Bandpass Filter.
    :param h_freq: The higher limit of the Bandpass Filter.
    :param ems_factor: This is a factor used for doing exponential moving standardization.
    :param init_block_size: This is the number of samples used to calculate the mean and standard deviation to apply
    the exponential moving standardization.
    :return: The preprocessed dataset.
    """

    def convert_from_volts_to_micro_volts(dataset: MOABBDataset = data) -> None:
        """
        This method converts an EEG Dataset from Volts to MicroVolts.
        :param dataset: The dataset which needs conversion from Volts to MicroVolts.
        :return: The converted dataset from Volts to MicroVolts.
        """
        return np.multiply(dataset, 1e6)

    # The following preprocessing is applied to the dataset,
    # 1. Keeps only the EEG Channel and drops MEG and STIM Channels.
    # 2. Converts the signals from Volts to MicroVolts. Hence, multiplies the received signal with a factor of 1e6.
    # 3. Use a Bandpass Filter to pass the signal between a certain range
    # 4. Apply exponential_moving_standardize with a factor of 1e-3 and init_block_size of 1000.
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(convert_from_volts_to_micro_volts),
        Preprocessor('filter', l_freq=l_freq, h_freq=h_freq),
        Preprocessor(exponential_moving_standardize, factor_new=ems_factor, init_block_size=init_block_size)
    ]
    return preprocess(data, preprocessors)
