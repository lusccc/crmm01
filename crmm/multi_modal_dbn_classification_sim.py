import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from learnergy.core import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.utils.data as data_utils

from learnergy.models.deep import DBN, ResidualDBN
from transformers import EvalPrediction

from metrics import calc_classification_metrics
from models.params_exposed_dbn import ParamsExposedDBN
from plygrnd.my_dbn_models import MyDBN

# Defining some input variables
batch_size = 1000
n_classes = 10
epoch = 20
fine_tune_epochs = 100
# use_gpu = False
model_type = 'sigmoid4deep'
use_gpu = True
use_add_fuse = True

# Creating training and validation/testing dataset
train_m1 = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_m1 = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

aug = [
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomRotation(degrees=(0, 180))
]
train_m2 = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(aug),
)
test_m2 = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(aug),
)

train_m1 = data_utils.Subset(train_m1, torch.arange(0, 1000))
train_m2 = data_utils.Subset(train_m2, torch.arange(0, 1000))
test_m1 = data_utils.Subset(test_m1, torch.arange(0, 200))
test_m2 = data_utils.Subset(test_m2, torch.arange(0, 200))

# Creating a DBN
model_m1 = DBN(
    model=(model_type, model_type),
    n_visible=784,
    n_hidden=(256, 256),
    steps=(1, 1),
    learning_rate=(0.1, 0.1),
    momentum=(0, 0),
    decay=(0, 0),
    temperature=(1, 1),
    use_gpu=use_gpu,
)
model_m2 = DBN(
    model=(model_type, model_type),
    n_visible=784,
    n_hidden=(256, 256),
    steps=(1, 1),
    learning_rate=(0.1, 0.1),
    momentum=(0, 0),
    decay=(0, 0),
    temperature=(1, 1),
    use_gpu=use_gpu,
)

if use_add_fuse:
    joint_n_vis = 256
else:
    joint_n_vis = 512

model_joint = DBN(
    model=(model_type, model_type),
    # n_visible=256,  # !!!
    n_visible=joint_n_vis,  # !!!
    n_hidden=(256, 256),
    steps=(1, 1),
    learning_rate=(0.1, 0.1),
    momentum=(0, 0),
    decay=(0, 0),
    temperature=(1, 1),
    use_gpu=use_gpu,
)

# Or you may create a ResidualDBN
# model = ResidualDBN(model='bernoulli', n_visible=784, n_hidden=(256, 256), steps=(1, 1),
#                     learning_rate=(0.1, 0.1), momentum=(0, 0), decay=(0, 0), temperature=(1, 1),
#                     zetta1=1, zetta2=1, use_gpu=True)

# Training a DBN

print('!!!! FIT MODEL_M1 !!!!')
_, m1_out, m1_targets, _ = model_m1.fit(train_m1, batch_size=batch_size, epochs=(epoch, epoch))
print('!!!! FIT MODEL_M2 !!!!')
_, m2_out, m2_targets, _ = model_m2.fit(train_m2, batch_size=batch_size, epochs=(epoch, epoch))

if use_add_fuse:
    fused_out = torch.add(m1_out, m2_out)
else:
    fused_out = torch.cat([m1_out, m2_out], dim=1)

fused_out = fused_out.clone().cpu().detach()
fused_out_dataset = Dataset(fused_out,
                            targets=torch.from_numpy(np.random.randint(2, size=len(fused_out))),
                            transform=lambda x: torch.from_numpy(x))  # note targets=feature is just placeholder
print('!!!! FIT MODEL_JOINT !!!!')
model_joint.fit(fused_out_dataset, batch_size=256, epochs=(epoch, epoch))

# Creating the Fully Connected layer to append on top of DBNs
fc = torch.nn.Linear(model_joint.n_hidden[model_joint.n_layers - 1], n_classes)

# Check if model uses GPU
if model_joint.device == "cuda":
    # If yes, put fully-connected on GPU
    fc = fc.cuda()

# Cross-Entropy loss is used for the discriminative fine-tuning
criterion = nn.CrossEntropyLoss()

# Creating the optimzers
# optimizer = []
optimizer = [optim.Adam(m.parameters(), lr=0.001) for m in model_m1.models] + \
            [optim.Adam(m.parameters(), lr=0.001) for m in model_m2.models] + \
            [optim.Adam(m.parameters(), lr=0.001) for m in model_joint.models]
optimizer.append(optim.Adam(fc.parameters(), lr=0.001))

# Creating training and validation batches
train_m1_batch = DataLoader(train_m1, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=8,
                            pin_memory=True, persistent_workers=True)
val_m1_batch = DataLoader(test_m1, batch_size=10000, shuffle=False, num_workers=8, prefetch_factor=8, pin_memory=True,
                          persistent_workers=True)

train_m2_batch = DataLoader(train_m2, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=8,
                            pin_memory=True, persistent_workers=True)
val_m2_batch = DataLoader(test_m2, batch_size=10000, shuffle=False, num_workers=8, prefetch_factor=8, pin_memory=True,
                          persistent_workers=True)

# For amount of fine-tuning epochs
for e in range(fine_tune_epochs):
    print(f"Epoch {e + 1}/{fine_tune_epochs}")

    # Resetting metrics
    train_loss, val_acc = 0, 0

    # For every possible batch
    for m1_batch, m2_batch in tqdm(zip(train_m1_batch, train_m2_batch)):
        # For every possible optimizer
        for opt in optimizer:
            # Resets the optimizer
            opt.zero_grad()

        # Flatenning the samples batch
        x_m1_batch, y_m1_batch = m1_batch
        x_m1_batch = x_m1_batch.reshape(x_m1_batch.size(0), model_m1.n_visible)

        x_m2_batch, y_m2_batch = m2_batch
        x_m2_batch = x_m2_batch.reshape(x_m2_batch.size(0), model_m2.n_visible)

        # Checking whether GPU is avaliable and if it should be used
        if model_joint.device == "cuda":
            # Applies the GPU usage to the data and labels
            x_m1_batch, y_m1_batch, x_m2_batch, y_m2_batch = \
                x_m1_batch.cuda(), \
                y_m1_batch.cuda(), \
                x_m2_batch.cuda(), \
                y_m2_batch.cuda()

        # Passing the batch down the model
        m1_out_batch = model_m1(x_m1_batch)
        m2_out_batch = model_m2(x_m2_batch)
        if use_add_fuse:
            fused = torch.add(m1_out_batch, m2_out_batch)
        else:
            fused = torch.cat([m1_out_batch, m2_out_batch], dim=1)
        y = model_joint(fused)

        # Calculating the fully-connected outputs
        y = fc(y)

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
    for m1_batch, m2_batch in tqdm(zip(val_m1_batch, val_m2_batch)):
        # Flatenning the samples batch
        x_m1_batch, y_m1_batch = m1_batch
        x_m1_batch = x_m1_batch.reshape(x_m1_batch.size(0), model_m1.n_visible)

        x_m2_batch, y_m2_batch = m2_batch
        x_m2_batch = x_m2_batch.reshape(x_m2_batch.size(0), model_m2.n_visible)

        # Checking whether GPU is avaliable and if it should be used
        if model_joint.device == "cuda":
            # Applies the GPU usage to the data and labels
            x_m1_batch, y_m1_batch, x_m2_batch, y_m2_batch = \
                x_m1_batch.cuda(), \
                y_m1_batch.cuda(), \
                x_m2_batch.cuda(), \
                y_m2_batch.cuda()

        # Passing the batch down the model
        m1_out_batch = model_m1(x_m1_batch)
        m2_out_batch = model_m2(x_m2_batch)
        if use_add_fuse:
            fused = torch.add(m1_out_batch, m2_out_batch)
        else:
            fused = torch.cat([m1_out_batch, m2_out_batch], dim=1)
        y = model_joint(fused)

        # Calculating the fully-connected outputs
        y = fc(y)

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

    # print(f"Loss: {train_loss / len(train_m1_batch)} | Val Accuracy: {val_acc}")

# Saving the fine-tuned model
# torch.save(model, "tuned_model.pth")
#
# # Checking the model's history
# for m in model.models:
#     print(m.history)
