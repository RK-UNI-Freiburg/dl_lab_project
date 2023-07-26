import shutil
import numpy as np
import neps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adagrad, RMSprop, AdamW

from src.networks import *
from src.utils import *
from src.training import *


def get_pipeline_space():
    pipeline_space = dict(
        lr=neps.FloatParameter(lower=0.0001, upper=0.001),
        bs=neps.IntegerParameter(lower=32, upper=80),
        o=neps.CategoricalParameter(choices=['sgd', 'adagrad', 'rmsprop', 'adamw']),
        nl=neps.IntegerParameter(lower=2, upper=8),
        nh=neps.CategoricalParameter(choices=[1, 2, 5, 10]),
        ies=neps.CategoricalParameter(choices=[10, 20, 30, 40, 50, 60]),
        hs=neps.IntegerParameter(lower=100, upper=200),
        dr=neps.FloatParameter(lower=0.3, upper=0.5),
    )
    return pipeline_space


def run_pipeline(lr, o, bs, nl, nh, ies, hs, dr):
    if o == 'adagrad':
        optimizer = Adagrad
    elif o == 'rmsprop':
        optimizer = RMSprop
    elif o == 'adamw':
        optimizer = AdamW
    else:
        optimizer = SGD

    results = train(
        exp_name=f'Conformer_LR_{lr}_O_{o}_BS_{bs}_NL_{nl}_NH_{nh}_IES_{ies}_HS_{hs}_DR_{dr}',
        torch_model=Conformer,
        num_epochs=200,
        batch_size=bs,
        learning_rate=lr,
        model_optimizer=optimizer,
        num_layers=nl,
        num_heads=nh,
        input_embedding_size=ies,
        hidden_size=hs,
        dropout=dr,
        return_results=True
    )
    return results


def run_bo():
    pipeline_space = get_pipeline_space()
    if os.path.exists("results/bayesian_optimization"):
        shutil.rmtree("results/bayesian_optimization")
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/bayesian_optimization",
        max_evaluations_total=30,
        searcher="bayesian_optimization",
    )
    previous_results, pending_configs = neps.status(
        "results/bayesian_optimization"
    )


if __name__ == '__main__':
    run_bo()
