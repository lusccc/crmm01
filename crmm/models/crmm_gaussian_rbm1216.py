import time
from typing import Optional, Tuple

import torch
from torch import nn
from torchviz import make_dot
from tqdm import tqdm
from transformers.utils import logging
import torch.nn.functional as F
from learnergy.models.gaussian import GaussianRBM4deep

logger = logging.get_logger('transformers')


class CrmmGaussianReluRBM4deep(GaussianRBM4deep):
    """A GaussianReluRBM class provides the basic implementation for
    Gaussian-ReLU Restricted Boltzmann Machines (for raw pixels values).

    Note that this class requires raw data (integer-valued)
    in order to model the image covariance into a latent ReLU layer.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

    """

    def __init__(
            self,
            n_visible: Optional[int] = 128,
            n_hidden: Optional[int] = 128,
            steps: Optional[int] = 1,
            learning_rate: Optional[float] = 0.001,
            momentum: Optional[float] = 0.0,
            decay: Optional[float] = 0.0,
            temperature: Optional[float] = 1.0,
            use_gpu: Optional[bool] = False,
            normalize: Optional[bool] = True,
            input_normalize: Optional[bool] = True,
            visible_layer=None,
    ) -> None:
        """Initialization method.

        Args:
            n_visible: Amount of visible units.
            n_hidden: Amount of hidden units.
            steps: Number of Gibbs' sampling steps.
            learning_rate: Learning rate.
            momentum: Momentum parameter.
            decay: Weight decay used for penalization.
            temperature: Temperature factor.
            use_gpu: Whether GPU should be used or not.
            normalize: Whether or not to use batch normalization.
            input_normalize: Whether or not to normalize inputs.

        """

        logger.info("Overriding class: GaussianRBM -> GaussianReluRBM.")

        # Override its parent class
        super(CrmmGaussianReluRBM4deep, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
            normalize,
            input_normalize,
        )

        self.visible_layer = None
        self.hidden_layer = None
        if visible_layer is not None:
            self.visible_layer = visible_layer

            self.W = None
            self.a = None
            self.b = None

            """used for visible sampling!"""
            self.hidden_layer = nn.Linear(self.n_hidden, self.n_visible)
        if normalize:
            # norm the visible layer output
            self.bn = nn.BatchNorm1d(self.n_hidden)


    def fit(
            self,
            samples,
            epochs: Optional[int] = 1,
            name=None
    ) -> Tuple[float, float]:
        """Fits a new SigmoidRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            (Tuple[float, float]): MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        """ 
        1. modification made to remove DataLoader and normalize in original learnergy!
        since it is directly computed in forward
        """
        # TODO make sure below
        """
        2. samples should be detach, because some input has required_grad=True, cause:
        `Trying to backward through the graph a second time`
        """
        samples = samples.detach()
        # print(f'\nbegin {name} rbm fit')
        # pbar = tqdm(range(epochs))
        for epoch in range(epochs):
            # print(f'epoch: {epoch}')
            start = time.time()
            mse = 0
            pl = 0

            # Performs the Gibbs sampling procedure
            _, _, _, _, visible_states = self.gibbs_sampling(samples)

            """comment this line, to make lower net layer take part in weight updating!???not sure"""
            # visible_states = visible_states.detach()

            cost = torch.mean(self.energy(samples)) - torch.mean(
                self.energy(visible_states)
            )

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            batch_size = samples.size(0)

            mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size
            ).detach()
            # pl = self.pseudo_likelihood(samples).detach()
            end = time.time()
            # pbar.set_description(f'fit rbm: {name}, cost: {cost}, mse: {mse}')
            # pbar.update()

            # self.dump(mse=mse.item(), pl=pl.item(), time=end - start)
        # print(f'end {name} rbm fit')
        return mse, None

    def hidden_sampling(
            self, v, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v: A tensor incoming from the visible layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the hidden layer sampling.

        """
        """ note is visible_layer!"""
        if self.visible_layer is None:
            activations = F.linear(v, self.W.t(), self.b)
        else:
            activations = self.visible_layer(v)
            if self.normalize:
                activations = self.bn(activations)
        if scale:
            probs = F.relu(torch.div(activations, self.T))
        else:
            probs = F.relu(activations)
        # Current states equals probabilities
        states = probs
        return probs, states

    def visible_sampling(
            self, h: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h: A tensor incoming from the hidden layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The probabilities and states of the visible layer sampling.

        """
        """note is hidden_layer!"""
        if self.hidden_layer is None:
            activations = F.linear(h, self.W, self.a)
        else:
            activations = self.hidden_layer(h)
        if scale:
            states = torch.div(activations, self.T)
        else:
            states = activations

        probs = torch.sigmoid(states)
        return probs, states

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculates and frees the system's energy.

        Args:
            samples: Samples to be energy-freed.

        Returns:
            (torch.Tensor): The system's energy based on input samples.

        """
        """ note is visible_layer!"""
        if self.visible_layer is None:
            activations = F.linear(samples, self.W.t(), self.b)
            a = self.a
        else:
            activations = self.visible_layer(samples)
            a = self.hidden_layer.bias
        # Creates a Softplus function for numerical stability
        s = nn.Softplus()

        h = torch.sum(s(activations), dim=1)
        v = 0.5 * torch.sum((samples - a) ** 2, dim=1)

        energy = v - h
        return energy
