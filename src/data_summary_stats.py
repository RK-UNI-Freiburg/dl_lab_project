import os
import argparse
import logging
import time
import numpy as np
import scipy as scp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

from .utils import *


def str_to_bool(value: str) -> bool:
    """
    This method converts boolean strings to boolean.
    :param value: String argument from command line.
    :return: Boolean based on the value field.
    """
    if value in ['true', 'True']:
        return True
    else:
        return False


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
                       init_block_size: int,
                       trial_start_offset_seconds: float) -> WindowsDataset:
    """
    This method does the required preprocessing on the dataset.
    :param data: This the dataset which has been fetched from MOABBDataset.
    :param l_freq: The lower limit of the Bandpass Filter.
    :param h_freq: The higher limit of the Bandpass Filter.
    :param ems_factor: This is a factor used for doing exponential moving standardization.
    :param init_block_size: This is the number of samples used to calculate the mean and standard deviation to apply
    the exponential moving standardization.
    :param trial_start_offset_seconds: This represents the duration (in seconds) before the event of interest starts.
    :return: The preprocessed and window cut dataset.
    """
    return dataset_preprocessor(data=data,
                                l_freq=l_freq,
                                h_freq=h_freq,
                                ems_factor=ems_factor,
                                init_block_size=init_block_size,
                                trial_start_offset_seconds=trial_start_offset_seconds)


def get_summary_stats(data_sets: Dict,
                      heatmap_flag: bool = True,
                      summary_stats_plots_path: str = 'plots/summary_statistics') -> None:
    """
    This method generates the summary statistics for the dataset at our hand.
    :param data_sets: This is the list of full training data, training data, validation data and the evaluation data.
    :param heatmap_flag: This flag decides if the channel wise correlation heatmaps are to be generated.
    :param summary_stats_plots_path: Location where the summary statistics plots are stored.
    :return: None
    """
    # Create a directory for the plots if it doesn't exist
    create_directory(summary_stats_plots_path)

    # The Exploratory Data Analysis is done for every dataset
    for name, data_set in data_sets.items():
        print(f'\nSummary stats for {name} ------------------------------------------------------------------\n')

        print(f'Number of Samples - {len(data_set)}')
        print(f'Shape of an Instance - {data_set[0][0].shape}')

        classes = np.array([instance[1] for instance in data_set])
        print(f'Class Distribution - \n{np.unique(classes, return_counts=True)}')

        data = np.array([instance[0] for instance in data_set])
        concatenated_data = np.concatenate(data, axis=1)
        if heatmap_flag:
            plt.figure(figsize=(20, 15))
            sns.heatmap(np.corrcoef(concatenated_data))
            plt.title(f"Channel Wise Correlation Map for {name}", fontdict={'fontsize': 30})
            plt.savefig(f"./{summary_stats_plots_path}/{name}_heatmap.png")
            plt.show()


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
    cmdline_parser.add_argument('-tsos', '--trial_start_offset_seconds',
                                default=-0.5,
                                help='This represents the duration (in seconds) before the event of interest starts',
                                type=float)
    cmdline_parser.add_argument('-tss', '--training_set_size',
                                default=0.8,
                                help='Provide the training set size in percentage, like 0.7 or 0.8',
                                type=float)
    cmdline_parser.add_argument('-hmf', '--heatmap_flag',
                                default=True,
                                help='If set to True, it will generate the channel wise correlation plots',
                                type=str_to_bool)
    cmdline_parser.add_argument('-sspp', '--summary_stats_plots_path',
                                default='plots/summary_statistics',
                                help='Location where the summary statistics plots are stored',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    # If the data does not exist, the fetch and store the data, else load the existing data
    if not any(os.scandir('./' + args.data_folder_name)):
        dataset = fetch_and_store(dataset_name=args.dataset_name,
                                  subject_ids=args.subject_ids,
                                  data_folder_name=args.data_folder_name)
    else:
        dataset = load_dataset(folder_name=args.data_folder_name)

    # Below, we preprocess the dataset we want to use for training purposes
    preprocessed_dataset = preprocess_dataset(data=dataset,
                                              l_freq=args.l_freq,
                                              h_freq=args.h_freq,
                                              ems_factor=args.ems_factor,
                                              init_block_size=args.init_block_size,
                                              trial_start_offset_seconds=args.trial_start_offset_seconds)

    # Below, we get the split datasets required for training purposes
    full_train_set, train_set, valid_set, eval_set = split_dataset(preprocessed_dataset,
                                                                   training_set_size=args.training_set_size)

    # Now we generate the summary statistics for the split datasets
    datasets = {
        'Full Training Set': full_train_set,
        'Training Set': train_set,
        'Validation Set': valid_set,
        'Evaluation Set': eval_set
    }
    get_summary_stats(datasets,
                      heatmap_flag=args.heatmap_flag,
                      summary_stats_plots_path=args.summary_stats_plots_path)
