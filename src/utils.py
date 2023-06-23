import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Any

from braindecode.datasets import MOABBDataset, WindowsDataset
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import exponential_moving_standardize, create_windows_from_events, \
    preprocess, Preprocessor

import torch
from torch.utils.data import Subset


def create_directory(folder_name: str) -> None:
    """
    This method creates a directory with desired name.
    :param folder_name: Name of the directory which is created.
    :return: None.
    """
    folder_path = './' + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


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


def conversions_and_filtering(data: MOABBDataset,
                              l_freq: float = 4.0,
                              h_freq: float = 38.0,
                              ems_factor: float = 1e-3,
                              init_block_size: int = 1000) -> MOABBDataset:
    """
    This method is responsible for applying all the required conversions and filtering as part of the preprocessing.
    :param data: The dataset which requires the conversions and filtering.
    :param l_freq: The lower limit of the Bandpass Filter.
    :param h_freq: The higher limit of the Bandpass Filter.
    :param ems_factor: This is a factor used for doing exponential moving standardization.
    :param init_block_size: This is the number of samples used to calculate the mean and standard deviation to apply
    the exponential moving standardization.
    :return: The dataset on which all the conversions and filtering has been applied.
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
    preprocess(data, preprocessors)
    return data


def cut_dataset_windows(data: MOABBDataset,
                        trial_start_offset_seconds: float = -0.5) -> WindowsDataset:
    """
    This method cuts windows of the dataset to generate individual instances for training purposes.
    :param data: The dataset on which the windows are cut out.
    :param trial_start_offset_seconds: This represents the duration (in seconds) before the event of interest starts.
    :return: The dataset which is cut into parts.
    """
    # s_freq contains the sampling frequency
    s_freq = data.datasets[0].raw.info['sfreq']  # The sampling frequency is 250 Hz
    trial_start_offset_samples = int(trial_start_offset_seconds * s_freq)
    data = create_windows_from_events(data,
                                      trial_start_offset_samples=trial_start_offset_samples,
                                      trial_stop_offset_samples=0,
                                      window_size_samples=None,
                                      window_stride_samples=None,
                                      drop_last_window=False,
                                      preload=True)
    return data


def dataset_preprocessor(data: MOABBDataset,
                         l_freq: float,
                         h_freq: float,
                         ems_factor: float,
                         init_block_size: int,
                         trial_start_offset_seconds: float) -> WindowsDataset:
    """
    This method applies all the required preprocessing to the MOABBDataset.
    :param data: The dataset which requires preprocessing.
    :param l_freq: The lower limit of the Bandpass Filter.
    :param h_freq: The higher limit of the Bandpass Filter.
    :param ems_factor: This is a factor used for doing exponential moving standardization.
    :param init_block_size: This is the number of samples used to calculate the mean and standard deviation to apply
    the exponential moving standardization.
    :param trial_start_offset_seconds: This represents the duration (in seconds) before the event of interest starts.
    :return: The preprocessed dataset.
    """
    # Apply all the required conversions and filtering required before preparing the training and validation datasets
    data = conversions_and_filtering(data, l_freq, h_freq, ems_factor, init_block_size)

    # Cutting windows of the dataset
    data = cut_dataset_windows(data, trial_start_offset_seconds)

    return data


def split_dataset(data: WindowsDataset, training_set_size: float = 0.8) -> Tuple[Any, Any, Any, Any]:
    """
    This method splits the dataset into full training data (train + validation), training data, validation data and
    the evaluation data.
    :param data: This is the dataset which is to be split into full data, training data, validation data and
    evaluation data.
    :param training_set_size: Indicates the training set size in percentage, like 0.7 or 0.8.
    :return: The full training data (train + validation), training data, validation data and the evaluation data.
    """
    split_data = data.split('session')  # Windowed dataset
    full_train_set = split_data['session_T']  # Training + Validation dataset
    split_index = int(len(full_train_set) * training_set_size)  # Index at which Training and Validation data is split
    train_set = Subset(full_train_set, range(0, split_index))  # Training dataset
    valid_set = Subset(full_train_set, range(split_index, 2592))  # Validation dataset
    eval_set = split_data['session_E']  # Evaluation dataset

    return full_train_set, train_set, valid_set, eval_set
