from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from learnergy.core import Dataset
from learnergy.models.bernoulli import ConvRBM
from learnergy.models.deep import ConvDBN, DBN
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import learnergy.utils.exception as e
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class DualConvDBNConcatClassifier(nn.Module):
    def __init__(self, dbn1, dbn2, n_classes) -> None:
        super().__init__()
        self.dbn1 = dbn1
        self.dbn2 = dbn2
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.fcs = nn.Sequential(
            nn.Linear(
                # hidden_shape is 2d
                np.prod(self.dbn1.models[-1].hidden_shape) * self.dbn1.n_filters[-1] +
                np.prod(self.dbn2.models[-1].hidden_shape) * self.dbn1.n_filters[-1],
                512),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes)
        )

    def forward(self, x1, x2):
        y1 = self.dbn1(x1)
        y1 = self.flatten1(y1)
        y2 = self.dbn2(x2)
        y2 = self.flatten2(y2)
        out = torch.cat([y1, y2], dim=1)
        out = self.fcs(out)
        return out


class ConvDBNJointClassifier(nn.Module):
    def __init__(self, dbn1, dbn2, joint_dbn, n_classes) -> None:
        super().__init__()
        self.dbn1 = dbn1
        self.dbn2 = dbn2
        self.joint_dbn = joint_dbn
        self.n_classes = n_classes
        self.flatten = nn.Flatten()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.joint_dbn.n_filters[-1], out_channels=8, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.fcs = nn.Sequential(
            # hidden_shape is 2d
            nn.Linear(in_features=128 * 55 * 55, out_features=512),
            nn.Dropout(.2),
            nn.ReLU(inplace=True),
            nn.Linear(512, 32),
            nn.Dropout(.1),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_classes)
        )
        # self.fcs = nn.Sequential(
        #     # hidden_shape is 2d
        #     nn.Linear(np.prod(self.joint_dbn.models[-1].hidden_shape) * self.joint_dbn.n_filters[-1], 512),
        #     nn.Dropout(.2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 128),
        #     nn.Dropout(.2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, n_classes)
        # )

    def forward(self, x1, x2):
        y1 = self.dbn1(x1)
        y2 = self.dbn2(x2)
        # joint_out = torch.cat([y1, y2], dim=1)
        out = torch.add(y1, y2)
        out = self.joint_dbn(out)
        out = self.convs(out)
        out = self.flatten(out)
        out = self.fcs(out)
        return out


class MyDBN(DBN):

    def __init__(self, model: Optional[str] = "bernoulli", n_visible: Optional[int] = 128,
                 n_hidden: Optional[Tuple[int, ...]] = (128,), steps: Optional[Tuple[int, ...]] = (1,),
                 learning_rate: Optional[Tuple[float, ...]] = (0.1,), momentum: Optional[Tuple[float, ...]] = (0.0,),
                 decay: Optional[Tuple[float, ...]] = (0.0,), temperature: Optional[Tuple[float, ...]] = (1.0,),
                 use_gpu: Optional[bool] = False):
        super().__init__(model, n_visible, n_hidden, steps, learning_rate, momentum, decay, temperature, use_gpu)
        self.model_list = nn.ModuleList(self.models)

    def fit(
            self,
            dataset: Union[torch.utils.data.Dataset, Dataset],
            batch_size: Optional[int] = 128,
            epochs: Optional[Tuple[int, ...]] = (10,),
    ) -> Tuple[float, float]:
        """Fits a new DBN model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs per layer.

        Returns:
            (Tuple[float, float]): MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        # Checking if the length of number of epochs' list is correct
        if len(epochs) != self.n_layers:
            # If not, raises an error
            raise e.SizeError(("`epochs` should have size equal as %d", self.n_layers))

        # Initializing MSE and pseudo-likelihood as lists
        mse, pl = [], []

        # Initializing the dataset's variables
        samples, targets, transform = (
            dataset.data.numpy(),
            dataset.targets.numpy(),
            dataset.transform,
        )

        # For every possible model (RBM)
        for i, model in enumerate(self.models):
            logger.info("Fitting layer %d/%d ...", i + 1, self.n_layers)

            # Creating the dataset
            d = Dataset(samples, targets, transform)

            # Fits the RBM
            model_mse, model_pl = model.fit(d, batch_size, epochs[i])

            # Appending the metrics
            mse.append(model_mse)
            pl.append(model_pl)

            # If the dataset has a transform
            if d.transform:
                # Applies the transform over the samples
                samples = d.transform(d.data)

            # If there is no transform
            else:
                # Just gather the samples
                samples = d.data

            # Checking whether GPU is avaliable and if it should be used
            if self.device == "cuda":
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Reshape the samples into an appropriate shape
            samples = samples.reshape(len(dataset), model.n_visible)

            # Gathers the targets
            targets = d.targets

            # Gathers the transform callable from current dataset
            transform = None

            # Performs a forward pass over the samples to get their probabilities
            samples, _ = model.hidden_sampling(samples)

            # Checking whether GPU is being used
            if self.device == "cuda":
                # If yes, get samples back to the CPU
                samples = samples.cpu()

            # Detaches the variable from the computing graph
            samples = samples.detach()

        """Modified to return `samples, targets, transform`"""
        return mse, samples, targets, transform

    def save_model(self, path):
        # multiple RBMs
        torch.save({f'model{i}': m.state_dict() for i, m in enumerate(self.models)}, path)

    def load_model(self, path):
        model_pt = torch.load(path)
        for i, m in enumerate(self.models):
            m.load_state_dict(model_pt[f'model{i}'])
