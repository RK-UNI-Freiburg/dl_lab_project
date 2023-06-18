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
    :return: The dataset fetched from MOABBDataset
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


def preprocess_dataset(data: MOABBDataset) -> None:
    """

    :param data: This the dataset which has been fetched from MOABBDataset.
    :return: The preprocessed dataset.
    """
    x = 5


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

    # preprocessed_dataset = preprocess_dataset(dataset)

    # get_summary_stats()
