import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.utils import logging

logger = logging.get_logger('transformers')


class CrmmRBM(nn.Module):
    """
    the visible layer and hidden layer can be custom models,
    and they are Continuous real values.
    Hence, our RBM is Gaussian-Gaussian RBM.
    """

    def __init__(self,
                 name=None,
                 visible_layer=None,
                 n_visible=128,
                 hidden_layer=None,
                 visible_output_normalize=True,
                 visible_input_normalize=True,
                 visible_output_dim=1,
                 steps=1,
                 temperature=1.0) -> None:
        super().__init__()
        self.name = name
        self.visible_layer = visible_layer
        self.hidden_layer = hidden_layer
        self.visible_output_normalize = visible_output_normalize
        self.visible_input_normalize = visible_input_normalize
        self.visible_output_dim = visible_output_dim
        self.steps = steps
        self.temperature = temperature

        self.visible_input_bn = nn.BatchNorm1d(self.n_visible) if self.visible_input_normalize else None
        self.visible_output_bn = nn.BatchNorm1d(self.n_hidden) if (
                self.visible_out_normalize and self.visible_out_dim == 1) else None

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)

    def fit(self, samples, epochs=1, name=None):
        # samples should be detached, because some input has required_grad=True, or it will cause:
        # `Trying to backward through the graph a second time`
        samples = samples.detach()
        for epoch in range(epochs):
            # Performs the Gibbs sampling procedure
            _, _, _, _, visible_states = self.gibbs_sampling(samples)
            visible_states = visible_states.detach()
            cost = torch.mean(self.energy(samples)) - torch.mean(self.energy(visible_states))

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_size = samples.size(0)
            mse = torch.div(torch.sum(torch.pow(samples - visible_states, 2)), batch_size).detach()
            print(f'{name}, cost: {cost}, mse: {mse}')

    def gibbs_sampling(self, v):
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v)
        neg_hidden_states = pos_hidden_states

        # Performing the Contrastive Divergence
        for step in range(self.steps):
            _, visible_states = self.visible_sampling(neg_hidden_states, True)
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, True
            )
        return (
            pos_hidden_probs,
            pos_hidden_states,
            neg_hidden_probs,
            neg_hidden_states,
            visible_states,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.hidden_sampling(x)
        return x

    def hidden_sampling(self, v, scale=False):
        """
        Performs the hidden layer sampling, i.e., P(h|v).
        """
        if self.visible_input_normalize:
            v = self.visible_input_bn(v)

        activations = self.visible_layer(v)

        if self.visible_output_normalize:
            activations = self.bn(activations)

        probs = F.relu(torch.div(activations, self.T) if scale else activations)
        # Current states equals probabilities
        states = probs
        return probs, states

    def visible_sampling(self, h, scale=False):
        """
        Performs the visible layer sampling, i.e., P(v|h).
        """
        activations = self.hidden_layer(h)
        states = torch.div(activations, self.T) if scale else activations
        probs = torch.sigmoid(states)
        return probs, states

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Calculates and frees the Gaussian-Gaussian RBM energy.
        paper 'A Gaussian-Gaussian-Restricted-Boltzmann-Machine-based Deep Neural Network Technique
        for Photovoltaic System Generation Forecasting'

        E(\mathbf{v},\mathbf{h})=
        \sum\limits_{i\in\mathbf{vis}}\dfrac{(v_i-a_i)^2}{2\sigma_i^2}+
        \sum\limits_{j\in\mathbf{hid}}\dfrac{(h_j-b_j)^2}{2\tau_j^2}-
        \sum\limits_{i,j}\dfrac{v_i}{\sigma_i}h_j w_{ij}
        """
        activations = self.visible_layer(samples)
        a = self.hidden_layer.bias
        # Creates a Softplus function for numerical stability
        s = nn.Softplus()

        h = torch.sum(s(activations), dim=1)
        v = 0.5 * torch.sum((samples - a) ** 2, dim=1)

        if h.shape != v.shape:
            # make shape same, do mean on v, because v may have more than 1 dim
            v = v.flatten(1).mean(1)

        energy = v - h
        return energy
