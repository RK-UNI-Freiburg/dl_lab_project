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
from sklearn.cluster import KMeans

from utils import *

import warnings
warnings.filterwarnings('ignore')


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


def get_summary_stats(data_sets: Dict,
                      n_clusters: int = 4,
                      summary_stats_plots_path: str = 'plots/summary_statistics') -> None:
    """
    This method generates the summary statistics for the dataset at our hand.
    :param data_sets: This is the list of full training data, training data, validation data and the evaluation data.
    :param n_clusters: Number of clusters to implement the channel wise clustering analysis.
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
        correlation_matrix = np.corrcoef(concatenated_data)

        # Plotting the channel wise correlation heatmaps
        plt.figure(figsize=(20, 15))
        sns.heatmap(correlation_matrix)
        plt.title(f"Channel Wise Correlation Map for {name}", fontdict={'fontsize': 30})
        plt.savefig(f"./{summary_stats_plots_path}/{name}_heatmap.png")
        plt.show()

        # Trying out KMeans Clustering to see which channels cluster together
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(correlation_matrix)
        cluster_labels = kmeans.labels_
        clusters = {}
        for cluster in range(n_clusters):
            clusters[cluster + 1] = []
        for channel, label in enumerate(cluster_labels):
            clusters[label + 1].append(channel + 1)
        print(f'Clustered Channels - \n{clusters}')
        x = np.linspace(1, 22, 22)
        y = cluster_labels + 1
        plt.figure(figsize=(15, 10))
        plt.scatter(x=x, y=y)
        plt.xlabel('Channel Number')
        plt.ylabel('Cluster Number')
        plt.title(f"Channel Wise Cluster Relation for {name}", fontdict={'fontsize': 30})
        plt.savefig(f"./{summary_stats_plots_path}/{name}_cluster_analysis.png")
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
    cmdline_parser.add_argument('-nc', '--n_clusters',
                                default=4,
                                help='Number of clusters to execute the channel wise clustering',
                                type=int)
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
    '''create_directory(args.data_folder_name)
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
                                              trial_start_offset_seconds=args.trial_start_offset_seconds)'''

    # Below, we get the split datasets required for training purposes after fetching and preprocessing it
    full_train_set, train_set, valid_set, eval_set = get_data_and_preprocess(dataset_dir=args.data_folder_name,
                                                                             dataset_name=args.dataset_name,
                                                                             subject_ids=args.subject_ids,
                                                                             l_freq=args.l_freq,
                                                                             h_freq=args.h_freq,
                                                                             ems_factor=args.ems_factor,
                                                                             init_block_size=args.init_block_size,
                                                                             trial_start_offset_seconds=args.trial_start_offset_seconds,
                                                                             training_set_size=args.training_set_size)

    # Now we generate the summary statistics for the split datasets
    datasets = {
        'Full Training Set': full_train_set,
        'Training Set': train_set,
        'Validation Set': valid_set,
        'Evaluation Set': eval_set
    }
    get_summary_stats(datasets,
                      n_clusters=args.n_clusters,
                      summary_stats_plots_path=args.summary_stats_plots_path)
