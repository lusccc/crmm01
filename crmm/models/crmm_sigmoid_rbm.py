"""Sigmoid-Bernoulli Restricted Boltzmann Machine.
"""

import time

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from learnergy.models.bernoulli import RBM
from transformers.utils import logging

logger = logging.get_logger('transformers')


class CRMMSigmoidRBM(RBM):
    """A SigmoidRBM class provides the basic implementation for
    Sigmoid-Bernoulli Restricted Boltzmann Machines.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

    """

    def __init__(
            self,
            n_visible: Optional[int] = 128,
            n_hidden: Optional[int] = 128,
            steps: Optional[int] = 1,
            learning_rate: Optional[float] = 0.1,
            momentum: Optional[float] = 0.0,
            decay: Optional[float] = 0.0,
            temperature: Optional[float] = 1.0,
            use_gpu: Optional[bool] = False,
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

        """

        logger.info("Overriding class: RBM -> SigmoidRBM.")

        super(CRMMSigmoidRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        logger.info("Class overrided.")

    def visible_sampling(
            self, h: torch.Tensor, scale: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h: A tensor incoming from the hidden layer.
            scale: A boolean to decide whether temperature should be used or not.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): The states and probabilities of the visible layer sampling.

        """

        activations = F.linear(h, self.W, self.a)

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        # Copies states as current probabilities
        states = probs

        return states, probs


class CRMMSigmoidRBM4deep(CRMMSigmoidRBM):
    """A SigmoidRBM class provides the basic implementation for
    Sigmoid-Bernoulli Restricted Boltzmann Machines.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

    """

    def __init__(
            self,
            n_visible: Optional[int] = 128,
            n_hidden: Optional[int] = 128,
            steps: Optional[int] = 1,
            learning_rate: Optional[float] = 0.1,
            momentum: Optional[float] = 0.0,
            decay: Optional[float] = 0.0,
            temperature: Optional[float] = 1.0,
            use_gpu: Optional[bool] = False,
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

        """

        logger.info("Overriding class: RBM -> SigmoidRBM.")

        super(CRMMSigmoidRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        logger.info("Class overrided.")

    def fit(
            self,
            samples,
            epochs: Optional[int] = 1,
    ) -> Tuple[float, float]:
        """Fits a new SigmoidRBM model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs.

        Returns:
            (Tuple[float, float]): MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        for epoch in range(epochs):
            start = time.time()
            mse = 0
            pl = 0

            # Performs the Gibbs sampling procedure
            _, _, _, _, visible_states = self.gibbs_sampling(samples)
            visible_states = visible_states.detach()

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
            pl = self.pseudo_likelihood(samples).detach()

            end = time.time()

            self.dump(mse=mse.item(), pl=pl.item(), time=end - start)

        return mse, pl