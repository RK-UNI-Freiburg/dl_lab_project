import os
import argparse
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import *


def fetch_and_store(dataset_name: str, subject_ids: str, data_folder_name: str) -> MOABBDataset:
    """
    This method fetches the required dataset and stores it.
    :param dataset_name: The name of the dataset which is to be fetched from braindecode MOABBDataset.
    :param subject_ids: The subjects whose data is fetched. This should be comma separated like 1,3,6 for subject ids
    1, 3 & 6.
    :param data_folder_name: The name of the folder where the preprocessed data is stored.
    :return: The dataset fetched from MOABBDataset.
    """
    # Create the folder where the data needs to be stored -------------------------------------------------------------
    create_directory(data_folder_name)

    # Fetch the dataset -----------------------------------------------------------------------------------------------
    subject_ids = str.split(subject_ids, ',')
    subject_ids = [int(num) for num in subject_ids]
    fetched_dataset = fetch_dataset(dataset_name, subject_ids)

    # Store the fetched dataset ---------------------------------------------------------------------------------------
    store_dataset(data_folder_name, fetched_dataset)

    return fetched_dataset


def preprocess_dataset(data: MOABBDataset,
                       l_freq: float,
                       h_freq: float,
                       ems_factor: float,
                       init_block_size: int) -> MOABBDataset:
    """
    This method does the required preprocessing on the dataset.
    :param data: This the dataset which has been fetched from MOABBDataset.
    :param l_freq: The lower limit of the Bandpass Filter.
    :param h_freq: The higher limit of the Bandpass Filter.
    :param ems_factor: This is a factor used for doing exponential moving standardization.
    :param init_block_size: This is the number of samples used to calculate the mean and standard deviation to apply
    the exponential moving standardization.
    :return: The preprocessed dataset.
    """
    return dataset_preprocessor(data=data,
                                l_freq= l_freq,
                                h_freq=h_freq,
                                ems_factor=ems_factor,
                                init_block_size=init_block_size)


def get_summary_stats() -> None:
    x = 5


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('Transformers for EEG')
    cmdline_parser.add_argument('-d', '--dataset_name',
                                default='BNCI2014001',
                                help='The name of the dataset to fetch',
                                type=str)
    cmdline_parser.add_argument('-s', '--subject_ids',
                                default='1,2,3,4,5,6,7,8,9',
                                help='Data of the Subject IDs to fetch',
                                type=str)
    cmdline_parser.add_argument('-dfn', '--data_folder_name',
                                default='data',
                                help='Folder name to store the data',
                                type=str)
    cmdline_parser.add_argument('-lf', '--l_freq',
                                default=4.0,
                                help='The lower limit of the Bandpass Filter',
                                type=float)
    cmdline_parser.add_argument('-hf', '--h_freq',
                                default=38.0,
                                help='The higher limit of the Bandpass Filter',
                                type=float)
    cmdline_parser.add_argument('-emsf', '--ems_factor',
                                default=1e-3,
                                help='Factor to apply exponential moving standardization',
                                type=float)
    cmdline_parser.add_argument('-ibs', '--init_block_size',
                                default=1000,
                                help='Initial block size to implement exponential moving standardization',
                                type=int)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if not any(os.scandir('./' + args.data_folder_name)):
        dataset = fetch_and_store(dataset_name=args.dataset_name,
                                  subject_ids=args.subject_ids,
                                  data_folder_name=args.data_folder_name)
    else:
        dataset = load_dataset(folder_name=args.data_folder_name)

    preprocessed_dataset = preprocess_dataset(data=dataset,
                                              l_freq=args.l_freq,
                                              h_freq=args.h_freq,
                                              ems_factor=args.ems_factor,
                                              init_block_size=args.init_block_size)

    # get_summary_stats()
