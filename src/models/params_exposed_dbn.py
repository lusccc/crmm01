from typing import Optional, Tuple, Union

import numpy
import torch
from learnergy.models.deep import DBN
from learnergy.models.deep import dbn
from learnergy.models.bernoulli import rbm
from learnergy.models.extra import sigmoid_rbm, SigmoidRBM4deep
import learnergy.utils.exception as e
from learnergy.core import model, Dataset
from torch import nn
# from transformers.utils import logging
from learnergy.models.bernoulli import RBM, DropoutRBM, EDropoutRBM
from learnergy.models.extra import SigmoidRBM
from learnergy.models.gaussian import (
    GaussianReluRBM,
    GaussianSeluRBM,
    VarianceGaussianRBM, GaussianRBM, GaussianRBM4deep, GaussianReluRBM4deep,
)
from torch.utils.data import DataLoader

from models.fix_gaussian_rbm import FixGaussianRBM, VarianceGaussianSeluRBM
from learnergy.utils import logging

logger = logging.get_logger(__name__)

# def setup_dbn_logger():
#     dbn.logger = logging.get_logger('transformers')
#     rbm.logger = logging.get_logger('transformers')
#     sigmoid_rbm.logger = logging.get_logger('transformers')
#     model.logger = logging.get_logger('transformers')

# MODELS = {
#     "bernoulli": RBM,
#     "dropout": DropoutRBM,
#     "e_dropout": EDropoutRBM,
#     "fix_gaussian": FixGaussianRBM,
#     "gaussian_relu": GaussianReluRBM,
#     "gaussian_selu": GaussianSeluRBM,
#     "sigmoid": SigmoidRBM,
#     "variance_gaussian": VarianceGaussianRBM,
#     "variance_selu_gaussian": VarianceGaussianSeluRBM,
#     "bernoulli": RBM,
#     "dropout": DropoutRBM,
#     "e_dropout": EDropoutRBM,
#     "gaussian": GaussianRBM,
#     "gaussian4deep": GaussianRBM4deep,
#     "gaussian_relu": GaussianReluRBM,
#     "gaussian_relu4deep": GaussianReluRBM4deep,
#     "gaussian_selu": GaussianSeluRBM,
#     "sigmoid": SigmoidRBM,
#     "sigmoid4deep": SigmoidRBM4deep,
#     "variance_gaussian": VarianceGaussianRBM,
# }
MODELS = {
    "bernoulli": RBM,
    "dropout": DropoutRBM,
    "e_dropout": EDropoutRBM,
    "gaussian": GaussianRBM,
    "gaussian4deep": GaussianRBM4deep,
    "gaussian_relu": GaussianReluRBM,
    "gaussian_relu4deep": GaussianReluRBM4deep,
    "gaussian_selu": GaussianSeluRBM,
    "sigmoid": SigmoidRBM,
    "sigmoid4deep": SigmoidRBM4deep,
    "variance_gaussian": VarianceGaussianRBM,
}


class ParamsExposedDBN(DBN):
    def __init__(self,
                 model: Optional[Tuple[str, ...]] = ("gaussian",),
                 n_visible: Optional[int] = 128,
                 n_hidden: Optional[Tuple[int, ...]] = (128,),
                 steps: Optional[Tuple[int, ...]] = (1,),
                 learning_rate: Optional[Tuple[float, ...]] = (0.1,),
                 momentum: Optional[Tuple[float, ...]] = (0.0,),
                 decay: Optional[Tuple[float, ...]] = (0.0,),
                 temperature: Optional[Tuple[float, ...]] = (1.0,),
                 use_gpu: Optional[bool] = False,
                 normalize: Optional[bool] = True,
                 input_normalize: Optional[bool] = True, ):
        # super().__init__(model, n_visible, n_hidden, steps, learning_rate, momentum, decay, temperature, use_gpu)
        # logger.info("Overriding class: Model -> DBN.")

        logger.info("Overriding class: Model -> DBN.")

        super(DBN, self).__init__(use_gpu=use_gpu)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_layers = len(n_hidden)

        self.steps = steps
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.T = temperature

        self.models = []
        model = list(model)
        if len(model) < self.n_layers:
            logger.info("\n\n> Incomplete number of RBMs, adding SigmoidRBMs to fill the stack!! <\n")
            for i in range(len(model) - 1, self.n_layers):
                model.append("sigmoid4deep")
        for i in range(self.n_layers):
            if i == 0:
                n_input = self.n_visible
            else:
                # Gathers the number of input units as previous number of hidden units
                n_input = self.n_hidden[i - 1]

                if model[i] == "sigmoid":
                    model[i] = "sigmoid4deep"
                elif model[i] == "gaussian":
                    model[i] = "gaussian4deep"
                elif model[i] == "gaussian_relu":
                    model[i] = "gaussian_relu4deep"
            try:
                m = MODELS[model[i]](
                    n_input,
                    self.n_hidden[i],
                    self.steps[i],
                    self.lr[i],
                    self.momentum[i],
                    self.decay[i],
                    self.T[i],
                    use_gpu,
                    normalize,
                    input_normalize,
                )

            except:
                m = MODELS[model[i]](
                    n_input,
                    self.n_hidden[i],
                    self.steps[i],
                    self.lr[i],
                    self.momentum[i],
                    self.decay[i],
                    self.T[i],
                    use_gpu,
                )
            self.models.append(m)
        model = tuple(model)

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")
        logger.debug("Number of layers: %d.", self.n_layers)

        self.model_list = nn.ModuleList(self.models)

    def fit(
            self,
            dataset: Union[torch.utils.data.Dataset, Dataset],
            batch_size: Optional[int] = 128,
            epochs: Optional[Tuple[int, ...]] = (10,),
    ) -> Tuple[float, float]:
        if len(epochs) != self.n_layers:
            raise e.SizeError(("`epochs` should have size equal as %d", self.n_layers))

        mse, pl = [], []

        try:
            samples, targets, transform = (
                dataset.data.numpy(),
                dataset.targets.numpy(),
                dataset.transform,
            )
            d = Dataset(samples, targets, transform)
        except:
            try:
                samples, targets, transform = (
                    dataset.data,
                    dataset.targets,
                    dataset.transform,
                )
                d = Dataset(samples, targets, transform)
            except:
                d = dataset

        batches = DataLoader(d, batch_size=batch_size, shuffle=True)

        for i, model in enumerate(self.models):
            logger.info("Fitting layer %d/%d ...", i + 1, self.n_layers)

            if i == 0:
                model_mse, model_pl = model.fit(d, batch_size, epochs[i])
                mse.append(model_mse)
                pl.append(model_pl)
            else:
                # creating the training phase for deeper models
                for ep in range(epochs[i]):
                    logger.info("Epoch %d/%d", ep + 1, epochs[i])
                    model_mse = 0
                    pl_ = 0
                    for step, (samples, y) in enumerate(batches):

                        samples = samples.reshape(len(samples), self.n_visible)

                        if self.device == "cuda":
                            samples = samples.cuda()

                        for ii in range(i):
                            samples, _ = self.models[ii].hidden_sampling(samples)

                        # Creating the dataset to ''mini-fit'' the i-th model
                        ds = Dataset(samples, y, None, show_log=False)
                        # Fiting the model with the batch
                        mse_, plh = model.fit(ds, samples.size(0), 1)
                        model_mse += mse_
                        pl_ += plh

                    model_mse /= len(batches)
                    pl_ /= len(batches)

                    # logger.info("MSE: %f", model_mse)
                    logger.info("MSE: %f | log-PL: %f", model_mse, pl_)
                mse.append(model_mse)
                pl.append(pl_)

        """Modified to return `samples, targets, transform`"""
        return mse, samples, targets, transform

    def save_model(self, path):
        # multiple RBMs
        torch.save({f'model{i}': m.state_dict() for i, m in enumerate(self.models)}, path)

    def load_model(self, path):
        model_pt = torch.load(path)
        for i, m in enumerate(self.models):
            m.load_state_dict(model_pt[f'model{i}'])
