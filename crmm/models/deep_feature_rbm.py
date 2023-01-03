from typing import Optional

import torch
from torch import nn

from learnergy.models.bernoulli import RBM
from models.crmm_gaussian_rbm import CrmmGaussianReluRBM4deep
from multimodal_transformers.model.layer_utils import MLP


class DeepFeatureRBM(CrmmGaussianReluRBM4deep):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.deep_nn = MLP(input_dim=self.n_visible,
                       output_dim=self.n_hidden,
                       num_hidden_lyr=3,
                       return_layer_outs=False,
                       bn=True)

    def hidden_sampling(
            self, v: torch.Tensor, scale: Optional[bool] = False
    ):
        probs = self.deep_nn(v)
        states = probs
        return probs, states

