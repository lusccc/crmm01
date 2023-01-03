import os.path
from typing import Tuple, Optional

import numpy
import torch
from learnergy.core import Dataset
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
dbn_epoch = 10
fine_tune_epoch = 30
batch_size = 300

num_emb = np.load(os.path.join(dbn_dataset_dir, f'predict_num_ae_embedding_results.npy'))
cat_emb = np.load(os.path.join(dbn_dataset_dir, f'predict_cat_ae_embedding_results.npy'))
labels = np.load(os.path.join(dbn_dataset_dir, f'labels.npy'))
num_emb_train, num_emb_val, num_emb_test = np.split(num_emb, [int(.8 * len(num_emb)), int(.9 * len(num_emb))])
cat_emb_train, cat_emb_val, cat_emb_test = np.split(cat_emb, [int(.8 * len(cat_emb)), int(.9 * len(cat_emb))])
labels_train, labels_val, labels_test = np.split(labels, [int(.8 * len(labels)), int(.9 * len(labels))])

tensor_transform_fn = lambda x: torch.from_numpy(x) if isinstance(x, numpy.ndarray) else x


def min_max_scale_transform_fn(x):
    x_min, x_max = x.min(), x.max()
    x = (x - x_min) / (x_max - x_min)
    return torch.from_numpy(x) if isinstance(x, numpy.ndarray) else x


num_emb_dt = Dataset(torch.from_numpy(num_emb),
                     targets=torch.from_numpy(labels),
                     transform=min_max_scale_transform_fn
                     )
num_emb_train_dt = Dataset(torch.from_numpy(num_emb_train),
                           targets=torch.from_numpy(labels_train),
                           transform=min_max_scale_transform_fn)
num_emb_val_dt = Dataset(torch.from_numpy(num_emb_val),
                         targets=torch.from_numpy(labels_val),
                         transform=min_max_scale_transform_fn)
num_emb_test_dt = Dataset(torch.from_numpy(num_emb_test),
                          targets=torch.from_numpy(labels_test),
                          transform=min_max_scale_transform_fn)

cat_emb_dt = Dataset(torch.from_numpy(cat_emb),
                     targets=torch.from_numpy(labels),
                     transform=min_max_scale_transform_fn)
cat_emb_train_dt = Dataset(torch.from_numpy(cat_emb_train),
                           targets=torch.from_numpy(labels_train),
                           transform=min_max_scale_transform_fn)
cat_emb_val_dt = Dataset(torch.from_numpy(cat_emb_val),
                         targets=torch.from_numpy(labels_val),
                         transform=min_max_scale_transform_fn)
cat_emb_test_dt = Dataset(torch.from_numpy(cat_emb_test),
                          targets=torch.from_numpy(labels_test),
                          transform=min_max_scale_transform_fn)
dbn_type = 'sigmoid'
# dbn_type = 'gaussian_relu'
# dbn_type = 'variance_selu_gaussian'
# dbn_type = 'fix_gaussian'
gbl_steps = 3
tmp = 1
num_dbn = ParamsExposedDBN(
    model=dbn_type,
    n_visible=64,
    n_hidden=(128, 128),
    steps=(gbl_steps, gbl_steps),
    learning_rate=(0.1, 0.1),
    momentum=(0, 0),
    decay=(0, 0),
    temperature=(tmp, tmp),
    use_gpu=use_gpu,
)

cat_dbn = ParamsExposedDBN(
    model=dbn_type,
    n_visible=64,
    n_hidden=(128, 128),
    steps=(gbl_steps, gbl_steps),
    learning_rate=(0.1, 0.1),
    momentum=(0, 0),
    decay=(0, 0),
    temperature=(tmp, tmp),
    use_gpu=use_gpu,
)
joint_dbn = ParamsExposedDBN(
    model=dbn_type,
    # n_visible=128,
    n_visible=128 + 128,
    n_hidden=(128, 128),
    steps=(gbl_steps, gbl_steps),
    learning_rate=(0.1, 0.1),
    momentum=(0, 0),
    decay=(0, 0),
    temperature=(tmp, tmp),
    use_gpu=use_gpu,
)

_, num_dbn_out, _, _ = num_dbn.fit(num_emb_dt, batch_size=batch_size, epochs=(dbn_epoch, dbn_epoch))
_, cat_dbn_out, _, _ = cat_dbn.fit(cat_emb_dt, batch_size=batch_size, epochs=(dbn_epoch, dbn_epoch))
fused_out = torch.cat([num_dbn_out, cat_dbn_out], dim=1)

"""把样本都错误分类为同一类别的原因：DBN的输入数据不在0-1的范围导致的,把joindbn的input手动加了batchnorm，结果就不行了"""
# fused_out = nn.BatchNorm1d(128+128, affine=False)(fused_out)
fused_out_dt = Dataset(fused_out,
                       targets=torch.from_numpy(np.random.randint(10, size=len(fused_out))),
                       transform=lambda x: torch.from_numpy(x))
joint_dbn.fit(fused_out_dt, batch_size=batch_size, epochs=(dbn_epoch, dbn_epoch))

n_classes = 10
fc = torch.nn.Linear(joint_dbn.n_hidden[joint_dbn.n_layers - 1], n_classes)
fc = fc.cuda() if use_gpu else fc

# below not work, will Twist the data!!
# bn1 = nn.BatchNorm1d(128+128)
# bn1 = bn1.cuda() if use_gpu else bn1
# bn2 = nn.BatchNorm1d(128)
# bn2 = bn2.cuda() if use_gpu else bn2

criterion = nn.CrossEntropyLoss()
optimizer = [optim.Adam(m.parameters(), lr=0.001) for m in num_dbn.models] + \
            [optim.Adam(m.parameters(), lr=0.001) for m in cat_dbn.models] + \
            [optim.Adam(m.parameters(), lr=0.001) for m in joint_dbn.models]
optimizer.append(optim.Adam(fc.parameters(), lr=0.001))
# optimizer.append(optim.Adam(bn1.parameters(), lr=0.001))
# optimizer.append(optim.Adam(bn2.parameters(), lr=0.001))

train_num_batch = DataLoader(num_emb_train_dt, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=8,
                             pin_memory=True, persistent_workers=True)
val_num_batch = DataLoader(num_emb_val_dt, batch_size=10000, shuffle=False, num_workers=8, prefetch_factor=8,
                           pin_memory=True,
                           persistent_workers=True)

train_cat_batch = DataLoader(cat_emb_train_dt, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=8,
                             pin_memory=True, persistent_workers=True)
val_cat_batch = DataLoader(cat_emb_val_dt, batch_size=10000, shuffle=False, num_workers=8, prefetch_factor=8,
                           pin_memory=True,
                           persistent_workers=True)

print('!!! fine tune')
for e in range(fine_tune_epoch):
    print(f"Epoch {e + 1}/{fine_tune_epoch}")

    # Resetting metrics
    train_loss, val_acc = 0, 0

    # For every possible batch
    for m1_batch, m2_batch in tqdm(zip(train_num_batch, train_cat_batch)):
        # For every possible optimizer
        for opt in optimizer:
            # Resets the optimizer
            opt.zero_grad()

        # Flatenning the samples batch
        x_m1_batch, y_m1_batch = m1_batch

        x_m2_batch, y_m2_batch = m2_batch

        # Checking whether GPU is avaliable and if it should be used
        if joint_dbn.device == "cuda":
            # Applies the GPU usage to the data and labels
            x_m1_batch, y_m1_batch, x_m2_batch, y_m2_batch = \
                x_m1_batch.cuda(), \
                y_m1_batch.cuda(), \
                x_m2_batch.cuda(), \
                y_m2_batch.cuda()

        # Passing the batch down the model
        m1_out_batch = num_dbn(x_m1_batch)
        m2_out_batch = cat_dbn(x_m2_batch)
        # fused = torch.add(m1_out_batch, m2_out_batch)
        fused = torch.cat([m1_out_batch, m2_out_batch], dim=1)
        # fused = bn1(fused)
        y = joint_dbn(fused)
        # y = bn2(y)
        # Calculating the fully-connected outputs
        y = fc(y)
        # y = F.selu(y)

        # Calculating loss
        loss = criterion(y, y_m1_batch)

        # Propagating the loss to calculate the gradients
        loss.backward()

        # For every possible optimizer
        for opt in optimizer:
            # Performs the gradient update
            opt.step()

        # Adding current batch loss
        train_loss += loss.item()

    # Calculate the test accuracy for the model:
    y_preds = []
    y_trues = []
    for m1_batch, m2_batch in tqdm(zip(val_num_batch, val_cat_batch)):
        x_m1_batch, y_m1_batch = m1_batch

        x_m2_batch, y_m2_batch = m2_batch

        # Checking whether GPU is avaliable and if it should be used
        if joint_dbn.device == "cuda":
            # Applies the GPU usage to the data and labels
            x_m1_batch, y_m1_batch, x_m2_batch, y_m2_batch = \
                x_m1_batch.cuda(), \
                y_m1_batch.cuda(), \
                x_m2_batch.cuda(), \
                y_m2_batch.cuda()

        # Passing the batch down the model
        m1_out_batch = num_dbn(x_m1_batch)
        m2_out_batch = cat_dbn(x_m2_batch)
        # fused = torch.add(m1_out_batch, m2_out_batch)
        fused = torch.cat([m1_out_batch, m2_out_batch], dim=1)
        # fused = bn1(fused)
        y = joint_dbn(fused)
        # y = bn2(y)
        # Calculating the fully-connected outputs
        y = fc(y)
        # y = F.selu(y)

        y_preds.append(y.detach().cpu().numpy())
        y_trues.append(y_m1_batch.detach().cpu().numpy())

        # # Calculating predictions
        # _, preds = torch.max(y, 1)
        #
        # # Calculating validation set accuracy
        # val_acc = torch.mean((torch.sum(preds == y_m1_batch).float()) / x_m1_batch.size(0))
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    ep = EvalPrediction(predictions=[y_preds, ], label_ids=y_trues, )
    metric = calc_classification_metrics(ep)
    print(f'metric: {metric}')
