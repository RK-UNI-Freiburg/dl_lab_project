import os
import argparse
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from src.networks import *
from src.utils import *


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


def main(exp_name: str,
         torch_model: Any,
         dataset_dir: str = './data',
         use_full_data: bool = False,
         num_epochs: int = 10,
         batch_size: int = 32,
         learning_rate: int = 0.0001,
         train_criterion: nn = nn.CrossEntropyLoss,
         model_optimizer: optim = optim.SGD,
         scheduler: optim = None,
         num_layers: int = 4,
         num_heads: int = 4,
         input_embedding_size: int = 1024,
         hidden_size: int = 512,
         dropout: int = 0.5,
         dataset_name: str = 'BNCI2014001',
         subject_ids: str = '1,2,3,4,5,6,7,8,9',
         l_freq: float = 4.0,
         h_freq: float = 38.0,
         ems_factor: float = 1e-3,
         init_block_size: int = 1000,
         trial_start_offset_seconds: float = -0.5,
         training_set_size: float = 0.8,
         model_path: str = 'models',
         plots_path: str = 'plots/training_results',
         test_preds_path: str = 'preds') -> None:
    """
    This method is a dynamic method which is used to train any architecture.
    :param exp_name: The name of the experiment being run.
    :param torch_model: The architecture class name which is used to train the model.
    :param dataset_dir: The name of the folder where the dataset is stored.
    :param use_full_data: If set to true uses full_train_set, else uses train_set
    :param num_epochs: Contains the number of epochs.
    :param batch_size: Contains the batch size.
    :param learning_rate: Contains the learning rate.
    :param train_criterion: Contains the loss criterion.
    :param model_optimizer: Contains the type of optimizer.
    :param scheduler: Contains the type of learning rate scheduler.
    :param num_layers: Contains the number of encoder layers for creating the transformer architecture.
    :param num_heads: Contains the number of heads for each self attention.
    :param input_embedding_size: Contains the embedding size of the input.
    :param hidden_size: Contains the hidden size of the feed forward part of the transformer architecture.
    :param dropout: Contains the level of dropout to be implemented
    :param dataset_name: The name of the dataset which is to be fetched from braindecode MOABBDataset.
    :param subject_ids: The subjects whose data is fetched.
    :param l_freq: The lower limit of the Bandpass Filter in data preprocessing.
    :param h_freq: The higher limit of the Bandpass Filter in data preprocessing.
    :param ems_factor: This is a factor used for doing exponential moving standardization in data preprocessing.
    :param init_block_size: This is the number of samples used to calculate the mean and standard deviation to apply
    the exponential moving standardization.
    :param trial_start_offset_seconds: This represents the duration (in seconds) before the event of interest starts.
    :param training_set_size: Indicates the training set size in percentage, like 0.7 or 0.8.
    :param model_path: The path where the pickle file of the model is stored.
    :param plots_path: The path where the training loss and accuracy plots are stored.
    :param test_preds_path: The path where the evaluation predictions are stored.
    :return: None
    """

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    create_directory(dataset_dir)
    if not any(os.scandir('./' + dataset_dir)):
        dataset = fetch_and_store(dataset_name=dataset_name,
                                  subject_ids=subject_ids,
                                  data_folder_name=dataset_dir)
    else:
        dataset = load_dataset(folder_name=dataset_dir)

    # Below, we preprocess the dataset we want to use for training purposes
    preprocessed_dataset = preprocess_dataset(data=dataset,
                                              l_freq=l_freq,
                                              h_freq=h_freq,
                                              ems_factor=ems_factor,
                                              init_block_size=init_block_size,
                                              trial_start_offset_seconds=trial_start_offset_seconds)

    # Below, we get the split datasets required for training purposes
    full_train_set, train_set, valid_set, eval_set = split_dataset(preprocessed_dataset,
                                                                   training_set_size=training_set_size)

    # Instantiating the Train, Validation and Test DataLoaders
    if use_full_data:
        train_loader = DataLoader(dataset=full_train_set,
                                  batch_size=batch_size,
                                  shuffle=True)
    else:
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True)
    val_loader = DataLoader(dataset=valid_set,
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(dataset=eval_set,
                             batch_size=batch_size,
                             shuffle=False)

    num_channels = full_train_set[0][0].shape[0]
    window_size = full_train_set[0][0].shape[1]
    num_classes = len(np.unique(np.array([instance[1] for instance in full_train_set])))
    input_shape = (num_channels, window_size)

    # Instantiate the model
    model = torch_model(num_layers=num_layers,
                        num_channels=num_channels,
                        num_heads=num_heads,
                        window_size=window_size,
                        input_embedding_size=input_embedding_size,
                        hidden_size=hidden_size,
                        dropout=dropout,
                        num_classes=num_classes).to(device)

    # Instantiating training criterion
    criterion = train_criterion().to(device)
    avg_train_loss = []
    avg_train_acc = []
    avg_val_loss = []
    avg_val_acc = []

    # Instantiate optimizer
    if model_optimizer == optim.SGD:
        optimizer = model_optimizer(model.parameters(), lr=learning_rate, momentum=0.9)
    elif model_optimizer == optim.Adagrad:
        optimizer = model_optimizer(model.parameters(), lr=learning_rate, lr_decay=0.3)
    elif model_optimizer == optim.RMSprop:
        optimizer = model_optimizer(model.parameters(), lr=learning_rate)
    elif model_optimizer == optim.Adam:
        optimizer = model_optimizer(model.parameters(), lr=learning_rate)
    elif model_optimizer == optim.AdamW:
        optimizer = model_optimizer(model.parameters(), lr=learning_rate)
    elif model_optimizer == optim.Adamax:
        optimizer = model_optimizer(model.parameters(), lr=learning_rate)
    elif model_optimizer == optim.Adadelta:
        optimizer = model_optimizer(model.parameters(), lr=learning_rate)
    else:
        optimizer = model_optimizer(model.parameters(), lr=learning_rate)

    # Instantiate scheduler
    if scheduler is not None:
        if scheduler == optim.lr_scheduler.LinearLR:
            scheduler = scheduler(optimizer)
        elif scheduler == optim.lr_scheduler.LambdaLR:
            scheduler = scheduler(optimizer)
        elif scheduler == optim.lr_scheduler.ExponentialLR:
            scheduler = scheduler(optimizer)
        elif scheduler == optim.lr_scheduler.CosineAnnealingLR:
            scheduler = scheduler(optimizer)
        elif scheduler == optim.lr_scheduler.CosineAnnealingWarmRestarts:
            scheduler = scheduler(optimizer, 3)
        elif scheduler == optim.lr_scheduler.StepLR:
            scheduler = scheduler(optimizer)
        else:
            scheduler = scheduler(optimizer)

    # Information on the model being trained
    # This provides the number of learnable parameters to be trained in the model
    logging.info('Model to be trained - ')
    summary(model, input_shape, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    # Loop through the epochs

    time_begin_full = time.time()

    for epoch in range(num_epochs):
        logging.info('#' * 70)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        time_begin = time.time()
        time_train = 0
        train_sum_loss = 0.0
        train_avg_loss = 0.0
        train_cnt_loss = 0
        train_sum_acc = 0.0
        train_avg_acc = 0.0
        train_cnt_acc = 0
        val_sum_loss = 0.0
        val_avg_loss = 0.0
        val_cnt_loss = 0
        val_sum_acc = 0.0
        val_avg_acc = 0.0
        val_cnt_acc = 0

        # Loop through the batches
        for i, data in enumerate(train_loader):
            signals = data[0].to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == labels) / len(labels)

            # Calculate the average loss and accuracy for all batches run so far
            n = signals.size(0)
            train_sum_loss += loss.item() * n
            train_cnt_loss += n
            train_avg_loss = train_sum_loss / train_cnt_loss
            train_sum_acc += acc.item() * n
            train_cnt_acc += n
            train_avg_acc = train_sum_acc / train_cnt_acc

        logging.info(f'Avg Train Loss: {train_avg_loss:.4f} | Avg Train Accuracy: {train_avg_acc * 100:.2f}%')
        avg_train_loss.append(train_avg_loss)
        avg_train_acc.append(train_avg_acc)

        for i, data in enumerate(val_loader):
            signals = data[0].to(device)
            labels = data[1].to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == labels) / len(labels)

            # Calculate the average loss and accuracy for all batches run so far
            n = signals.size(0)
            val_sum_loss += loss.item() * n
            val_cnt_loss += n
            val_avg_loss = val_sum_loss / val_cnt_loss
            val_sum_acc += acc.item() * n
            val_cnt_acc += n
            val_avg_acc = val_sum_acc / val_cnt_acc

        logging.info(f'Avg Valid Loss: {val_avg_loss:.4f} | Avg Valid Accuracy: {val_avg_acc * 100:.2f}%')
        avg_val_loss.append(val_avg_loss)
        avg_val_acc.append(val_avg_acc)
        time_train += time.time() - time_begin
        logging.info(f'Training Time: {time_train / 60:.2f} Minutes')

    time_train_full = time.time() - time_begin_full
    logging.info(f'Overall Training Time: {time_train_full / 60:.2f} Minutes')
    logging.info(f'Average Training Time / Epoch: {time_train_full / (60 * num_epochs):.2f} Minutes')

    # Generate predictions for test set
    predictions = np.array([])
    eval_sum_loss = 0.0
    eval_avg_loss = 0.0
    eval_cnt_loss = 0
    eval_sum_acc = 0.0
    eval_avg_acc = 0.0
    eval_cnt_acc = 0
    for i, data in enumerate(test_loader):
        signals = data[0].to(device)
        labels = data[1].to(device)
        logits = model(signals)
        preds = torch.argmax(logits, dim=1)
        predictions = np.append(predictions, preds.cpu().detach().numpy())
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels) / len(labels)

        # Calculate the average loss and accuracy for all batches run so far
        n = signals.size(0)
        eval_sum_loss += loss.item() * n
        eval_cnt_loss += n
        eval_avg_loss = eval_sum_loss / eval_cnt_loss
        eval_sum_acc += acc.item() * n
        eval_cnt_acc += n
        eval_avg_acc = eval_sum_acc / eval_cnt_acc

    logging.info(f'Evaluation Loss: {eval_avg_loss:.4f} | Evaluation Accuracy: {eval_avg_acc * 100:.2f}%')
    create_directory(test_preds_path)
    np.savetxt(os.path.join('./' + test_preds_path, exp_name + '_' + str(int(time.time()))) + '.csv', predictions)
    logging.info(f'Saved evaluation predictions in ./{test_preds_path} folder')

    # Plot the Losses and Accuracies for Training and Validation
    if plots_path:
        create_directory(plots_path)
        x = range(1, num_epochs + 1)
        fig = plt.figure(figsize=(10, 12))
        fig.suptitle(exp_name + ' Plots')
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(x, avg_train_loss)
        ax1.plot(x, avg_val_loss)
        ax1.set_title('Avg Loss Plot')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend(['Training Loss', 'Validation Loss'])

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(x, [acc * 100 for acc in avg_train_acc])
        ax2.plot(x, [acc * 100 for acc in avg_val_acc])
        ax2.set_title('Avg Accuracy Plot')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (in %age)')
        ax2.legend(['Training Accuracy', 'Validation Accuracy'])

        fig.savefig(os.path.join('./' + plots_path, exp_name + '_' + str(int(time.time()))) + '.png')
        logging.info(f'Saved plots in ./{plots_path} folder')

    if model_path:
        create_directory(model_path)
        torch.save(model.state_dict(), os.path.join('./' + model_path, exp_name + '_model_' + str(int(time.time()))))
        logging.info(f'Saved model in ./{model_path} folder')

    logging.info('Training and Evaluation Completed!')


if __name__ == '__main__':

    loss_dict = {
        'cross_entropy': nn.CrossEntropyLoss
    }

    opti_dict = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'rms': optim.RMSprop,
        'ada_grad': optim.Adagrad,
        'adam_w': optim.AdamW,
        'ada_max': optim.Adamax,
        'ada_delta': optim.Adadelta
    }

    lr_scheduler = {
        'linear': optim.lr_scheduler.LinearLR,
        'lambda': optim.lr_scheduler.LambdaLR,
        'exp': optim.lr_scheduler.ExponentialLR,
        'cosine': optim.lr_scheduler.CosineAnnealingLR,
        'warm': optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'step': optim.lr_scheduler.StepLR,
        'none': None
    }

    cmdline_parser = argparse.ArgumentParser('Train Models')
    cmdline_parser.add_argument('-n', '--exp_name',
                                default='default',
                                help='Name of this experiment',
                                type=str)
    cmdline_parser.add_argument('-m', '--model',
                                default='EEGTransformer',
                                help='Class name of model to train',
                                type=str)
    cmdline_parser.add_argument('-dd', '--dataset_dir',
                                default='data',
                                help='Path to the dataset to use for training',
                                type=str)
    cmdline_parser.add_argument('-ufd', '--use_full_data',
                                default=False,
                                help='Flag to decide if we need full data for training',
                                type=str_to_bool)
    cmdline_parser.add_argument('-e', '--epochs',
                                default=10,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-bs', '--batch_size',
                                default=32,
                                help='Batch Size',
                                type=int)
    cmdline_parser.add_argument('-lr', '--learning_rate',
                                default=0.0001,
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-tl', '--training_loss',
                                default='cross_entropy',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='sgd',
                                help='Which optimizer to use during training',
                                choices=list(opti_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-lrs', '--lr_scheduler',
                                default='none',
                                help='Which learning rate scheduler to use during training',
                                choices=list(lr_scheduler.keys()),
                                type=str)
    cmdline_parser.add_argument('-nl', '--num_layers',
                                default=4,
                                help='Number of Layers',
                                type=int)
    cmdline_parser.add_argument('-nh', '--num_heads',
                                default=4,
                                help='Number of Heads',
                                type=int)
    cmdline_parser.add_argument('-ies', '--input_embedding_size',
                                default=1024,
                                help='Input Embedding Size',
                                type=int)
    cmdline_parser.add_argument('-hs', '--hidden_size',
                                default=512,
                                help='Hidden Size',
                                type=int)
    cmdline_parser.add_argument('-dr', '--dropout',
                                default=0.5,
                                help='Dropout',
                                type=int)
    cmdline_parser.add_argument('-dn', '--dataset_name',
                                default='BNCI2014001',
                                help='The name of the dataset to fetch',
                                type=str)
    cmdline_parser.add_argument('-sid', '--subject_ids',
                                default='1,2,3,4,5,6,7,8,9',
                                help='Data of the Subject IDs to fetch',
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
    cmdline_parser.add_argument('-mp', '--model_path',
                                default='models',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-pp', '--plots_path',
                                default='plots/training_results',
                                help='Path to store loss and accuracy graphs',
                                type=str)
    cmdline_parser.add_argument('-tpp', '--test_preds_path',
                                default='preds',
                                help='Path to store test set predictions',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    main(exp_name=args.exp_name,
         torch_model=eval(args.model),
         dataset_dir=args.dataset_dir,
         use_full_data=args.use_full_data,
         num_epochs=args.epochs,
         batch_size=args.batch_size,
         learning_rate=args.learning_rate,
         train_criterion=loss_dict[args.training_loss],
         model_optimizer=opti_dict[args.optimizer],
         scheduler=lr_scheduler[args.lr_scheduler],
         num_layers=args.num_layers,
         num_heads=args.num_heads,
         input_embedding_size=args.input_embedding_size,
         hidden_size=args.hidden_size,
         dropout=args.dropout,
         dataset_name=args.dataset_name,
         subject_ids=args.subject_ids,
         l_freq=args.l_freq,
         h_freq=args.h_freq,
         ems_factor=args.ems_factor,
         init_block_size=args.init_block_size,
         trial_start_offset_seconds=args.trial_start_offset_seconds,
         training_set_size=args.training_set_size,
         model_path=args.model_path,
         plots_path=args.plots_path,
         test_preds_path=args.test_preds_path)
