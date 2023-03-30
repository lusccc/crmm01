import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchviz import make_dot
from tqdm import tqdm
import numpy as np
from learnergy.models.deep import DBN, ResidualDBN
from transformers import EvalPrediction

from metrics import calc_classification_metrics
from plygrnd.my_dbn_models import MyDBN

torch.manual_seed(3407)
# Defining some input variables
batch_size = 128
n_classes = 10
fine_tune_epochs = 20

# Creating training and validation/testing dataset
train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Creating a DBN
model = MyDBN(
    model="sigmoid",
    n_visible=784,
    n_hidden=(256, 256),
    steps=(1, 1),
    learning_rate=(0.1, 0.1),
    momentum=(0, 0),
    decay=(0, 0),
    temperature=(1, 1),
    use_gpu=True,
)

# Or you may create a ResidualDBN
# model = ResidualDBN(model='bernoulli', n_visible=784, n_hidden=(256, 256), steps=(1, 1),
#                     learning_rate=(0.1, 0.1), momentum=(0, 0), decay=(0, 0), temperature=(1, 1),
#                     zetta1=1, zetta2=1, use_gpu=True)


# Training a DBN
_, out, target, _ = model.fit(train, batch_size=batch_size, epochs=(5, 5))

# x_train_tensors = list(DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0))[0][0]
# x_train_tensors = torch.reshape(x_train_tensors, (-1, 28*28)).cuda()
# out_test = model(x_train_tensors)
# Creating the Fully Connected layer to append on top of DBNs
fc = torch.nn.Linear(model.n_hidden[model.n_layers - 1], n_classes)

# Check if model uses GPU
if model.device == "cuda":
    # If yes, put fully-connected on GPU
    fc = fc.cuda()

# Cross-Entropy loss is used for the discriminative fine-tuning
criterion = nn.CrossEntropyLoss()

# Creating the optimzers
# optimizer = []
optimizer = [optim.Adam(m.parameters(), lr=0.001) for m in model.models]
optimizer.append(optim.Adam(fc.parameters(), lr=0.001))

# Creating training and validation batches
train_batch = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=1)
val_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=1)

# For amount of fine-tuning epochs
for e in range(fine_tune_epochs):
    print(f"Epoch {e + 1}/{fine_tune_epochs}")

    # Resetting metrics
    train_loss, val_acc = 0, 0

    # For every possible batch
    for x_batch, y_batch in tqdm(train_batch):
        # For every possible optimizer
        for opt in optimizer:
            # Resets the optimizer
            opt.zero_grad()

        # Flatenning the samples batch
        x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

        # Checking whether GPU is avaliable and if it should be used
        if model.device == "cuda":
            # Applies the GPU usage to the data and labels
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        # Passing the batch down the model
        y = model(x_batch)

        # Calculating the fully-connected outputs
        y = fc(y)

        # Calculating loss
        loss = criterion(y, y_batch)

        make_dot(loss, show_attrs=True, ).render("dbnclf", directory='./', format="pdf")

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
    for x_batch, y_batch in tqdm(val_batch):
        # Flatenning the testing samples batch
        x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

        # Checking whether GPU is avaliable and if it should be used
        if model.device == "cuda":
            # Applies the GPU usage to the data and labels
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        # Passing the batch down the model
        y = model(x_batch)

        # Calculating the fully-connected outputs
        y = fc(y)

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

    print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}")

# Saving the fine-tuned model
torch.save(model, "tuned_model.pth")

# Checking the model's history
for m in model.models:
    print(m.history)
