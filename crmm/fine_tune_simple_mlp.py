import os
import torch
from torch import nn, optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from transformers import EvalPrediction

from metrics import calc_classification_metrics


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

batch_size=2000
fine_tune_epochs = 1000
use_gpu = True
dbn_dataset_dir = 'data/cr_sec_ae_embedding'
num_emb = np.load(os.path.join(dbn_dataset_dir, f'predict_num_ae_embedding_results.npy'))
cat_emb = np.load(os.path.join(dbn_dataset_dir, f'predict_cat_ae_embedding_results.npy'))
labels = np.load(os.path.join(dbn_dataset_dir, f'labels.npy'))
# labels = labels.reshape((-1, 1))

# fused_emb = num_emb + cat_emb
fused_emb = np.hstack([num_emb,cat_emb])
emb_train, emb_val, emb_test = np.split(fused_emb, [int(.8 * len(fused_emb)), int(.9 * len(fused_emb))])
labels_train, labels_val, labels_test = np.split(labels, [int(.8 * len(labels)), int(.9 * len(labels))])

train_dt = TensorDataset(torch.from_numpy(emb_train), torch.from_numpy(labels_train))
val_dt = TensorDataset(torch.from_numpy(emb_val), torch.from_numpy(labels_val))
test_dt = TensorDataset(torch.from_numpy(emb_test), torch.from_numpy(labels_test))

train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=8,
                             pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dt, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=8,
                             pin_memory=True, persistent_workers=True)

model = MLP()
model = model.cuda() if use_gpu else model
opt = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for e in range(fine_tune_epochs):
    print(f"Epoch {e+1}/{fine_tune_epochs}")

    # Resetting metrics
    train_loss, val_acc = 0, 0

    # For every possible batch
    for x_batch, y_batch in tqdm(train_loader):
        # For every possible optimizer
        opt.zero_grad()

        # Checking whether GPU is avaliable and if it should be used
        if use_gpu:
            # Applies the GPU usage to the data and labels
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        # Passing the batch down the model
        y = model(x_batch)


        # Calculating loss
        loss = criterion(y, y_batch)


        # Propagating the loss to calculate the gradients
        loss.backward()

        opt.step()

        # Adding current batch loss
        train_loss += loss.item()

    # Calculate the test accuracy for the model:
    y_preds = []
    y_trues = []
    for x_batch, y_batch in tqdm(val_loader):
        # Checking whether GPU is avaliable and if it should be used
        if use_gpu:
            # Applies the GPU usage to the data and labels
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        # Passing the batch down the model
        y = model(x_batch)

        y_preds.append(y.detach().cpu().numpy())
        y_trues.append(y_batch.detach().cpu().numpy())

        # Calculating predictions
        # _, preds = torch.max(y, 1)

        # Calculating validation set accuracy
        # val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    ep = EvalPrediction(predictions=[y_preds, ], label_ids=y_trues, )
    metric = calc_classification_metrics(ep)
    print(f'metric: {metric}')

    print(f"Loss: {train_loss / len(train_loader)} | Val Accuracy: {val_acc}")


