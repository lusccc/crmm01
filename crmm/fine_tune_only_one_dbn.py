import os.path
from typing import Tuple, Optional

import numpy
import torch
from learnergy.core import Dataset
from learnergy.models.deep import DBN
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import PreTrainedModel, PretrainedConfig, EvalPrediction

import analysis
from metrics import calc_classification_metrics
from models.params_exposed_dbn import ParamsExposedDBN
from multimodal_transformers.model.layer_utils import calc_mlp_dims, MLP, hf_loss_func
import numpy as np
from transformers.utils import logging
from utils import utils

logger = logging.get_logger('transformers')

use_gpu = True
# use_gpu = False
dbn_dataset_dir = 'data/cr_sec_ae_embedding'
dbn_epoch = 100
fine_tune_epoch = 200
batch_size = 2000

num_emb = np.load(os.path.join(dbn_dataset_dir, f'predict_num_ae_embedding_results.npy'))
labels = np.load(os.path.join(dbn_dataset_dir, f'labels.npy'))
num_emb_train, num_emb_val, num_emb_test = np.split(num_emb, [int(.8 * len(num_emb)), int(.9 * len(num_emb))])
labels_train, labels_val, labels_test = np.split(labels, [int(.8 * len(labels)), int(.9 * len(labels))])

tensor_transform_fn = lambda x: torch.from_numpy(x) if isinstance(x, numpy.ndarray) else x


def min_max_scale_transform_fn(x):
    x = tensor_transform_fn(x)
    x_min, x_max = x.min(), x.max()
    x = (x - x_min) / (x_max - x_min)
    #!!!!!!!!!! TODO
    return x

def standard_scale_transform_fn(x):
    x = tensor_transform_fn(x)
    mean, std, var = torch.mean(x), torch.std(x), torch.var(x)
    x = (x - mean) / std
    return x


def sigmoid_transform(x):
    x = tensor_transform_fn(x)
    return nn.Sigmoid()(x)


transform_fn = tensor_transform_fn

num_emb_dt = Dataset(torch.from_numpy(num_emb),
                     targets=torch.from_numpy(labels),
                     transform=transform_fn
                     )
num_emb_train_dt = Dataset(torch.from_numpy(num_emb_train),
                           targets=torch.from_numpy(labels_train),
                           transform=transform_fn)
num_emb_val_dt = Dataset(torch.from_numpy(num_emb_val),
                         targets=torch.from_numpy(labels_val),
                         transform=transform_fn)
num_emb_test_dt = Dataset(torch.from_numpy(num_emb_test),
                          targets=torch.from_numpy(labels_test),
                          transform=transform_fn)

dbn_type = 'sigmoid'
# dbn_type = 'sigmoid4deep'
# dbn_type = 'sigmoid4deep'
# dbn_type = 'variance_selu_gaussian'
# dbn_type = 'fix_gaussian'
gbl_steps = 1
tmp = 1
num_dbn = DBN(
    model=(dbn_type, dbn_type),
    n_visible=64,
    n_hidden=(100, 64),
    steps=(gbl_steps, gbl_steps),
    learning_rate=(0.001, 0.001),
    momentum=(0,0),
    decay=(0,0),
    temperature=(tmp, tmp),
    use_gpu=use_gpu,
)
# _, num_dbn_out, _, _ =
num_dbn.fit(num_emb_dt, batch_size=batch_size, epochs=(dbn_epoch, dbn_epoch))

n_classes = 10
fc = nn.Sequential(
    nn.Linear(num_dbn.n_hidden[num_dbn.n_layers - 1], 64),
    # nn.ReLU(),
    # nn.Linear(64, 128),
    # nn.ReLU(),
    # nn.Linear(128, 64),
    # nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, n_classes),
    nn.ReLU()
)
# fc = nn.Linear(num_dbn.n_hidden[num_dbn.n_layers - 1], n_classes)
fc = fc.cuda() if use_gpu else fc

complete_model = nn.Sequential(
    num_dbn,
    fc
)
optimizer = optim.Adam(complete_model.parameters())

criterion = nn.CrossEntropyLoss()
# optimizer = [optim.Adam(m.parameters(), lr=0.001) for m in num_dbn.models]
# optimizer.append(optim.Adam(fc.parameters(), lr=0.001))

train_num_batch = DataLoader(num_emb_train_dt, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=8,
                             pin_memory=True, persistent_workers=True)
val_num_batch = DataLoader(num_emb_val_dt, batch_size=10000, shuffle=False, num_workers=8, prefetch_factor=8,
                           pin_memory=True,
                           persistent_workers=True)

print('!!! fine tune')
for e in range(fine_tune_epoch):
    print(f"Epoch {e + 1}/{fine_tune_epoch}")

    # Resetting metrics
    train_loss, val_acc = 0, 0

    # For every possible batch
    for m1_batch in tqdm(train_num_batch):
        # For every possible optimizer
        # for opt in optimizer:
        #     # Resets the optimizer
        #     opt.zero_grad()
        optimizer.zero_grad()

        # Flatenning the samples batch
        x_m1_batch, y_m1_batch = m1_batch

        # Checking whether GPU is avaliable and if it should be used
        if num_dbn.device == "cuda":
            # Applies the GPU usage to the data and labels
            x_m1_batch, y_m1_batch = \
                x_m1_batch.cuda(), \
                y_m1_batch.cuda()

        # y = num_dbn(x_m1_batch)
        # y = fc(y)
        y = complete_model(x_m1_batch)

        # Calculating loss
        loss = criterion(y, y_m1_batch)

        # Propagating the loss to calculate the gradients
        loss.backward()

        # For every possible optimizer
        # for opt in optimizer:
        #     # Performs the gradient update
        #     opt.step()
        optimizer.step()

        # Adding current batch loss
        train_loss += loss.item()

    # Calculate the test accuracy for the model:
    y_preds = []
    y_trues = []
    for m1_batch in tqdm(val_num_batch):
        x_m1_batch, y_m1_batch = m1_batch

        # Checking whether GPU is avaliable and if it should be used
        if num_dbn.device == "cuda":
            # Applies the GPU usage to the data and labels
            x_m1_batch, y_m1_batch = \
                x_m1_batch.cuda(), \
                y_m1_batch.cuda()

        # y = num_dbn(x_m1_batch)
        # y = fc(y)
        y = complete_model(x_m1_batch)

        y_preds.append(y.detach().cpu().numpy())
        y_trues.append(y_m1_batch.detach().cpu().numpy())

        # # Calculating predictions
        _, preds = torch.max(y, 1)
        a = 1
        #
        # # Calculating validation set accuracy
        # val_acc = torch.mean((torch.sum(preds == y_m1_batch).float()) / x_m1_batch.size(0))
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    ep = EvalPrediction(predictions=[y_preds, ], label_ids=y_trues, )
    metric = calc_classification_metrics(ep)
    print(f'metric: {metric}')
