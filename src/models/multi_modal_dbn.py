import json
import os.path
from typing import Tuple, Optional

import numpy
import torch
from torchviz import make_dot

from learnergy.core import Dataset
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import PreTrainedModel, PretrainedConfig, EvalPrediction

import analysis
from learnergy.models.deep import DBN
from metrics import calc_classification_metrics
from models.crmm_dbn import CRMMDBN
from models.params_exposed_dbn import ParamsExposedDBN
from multimodal_transformers.model.layer_utils import calc_mlp_dims, MLP, hf_loss_func
import numpy as np
from transformers.utils import logging
from utils import utils

logger = logging.get_logger('transformers')


class MultiModalDBNConfig(PretrainedConfig):
    model_type = 'multi_modal_dbn'

    modality_list = ('num', 'cat')

    def __init__(self,
                 tabular_config=None,
                 use_modality=modality_list,
                 dbn_train_epoch=2,
                 use_gpu=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.tabular_config = tabular_config
        self.n_modality = len(use_modality)
        self.use_modality = {
            'num': True if 'num' in use_modality else False,
            'cat': True if 'cat' in use_modality else False,
        }
        self.dbn_train_epoch = dbn_train_epoch
        self.use_gpu = use_gpu

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, default=lambda o: o.__dict__, indent=2, sort_keys=True) + "\n"


class MultiModalDBN(PreTrainedModel):
    def __init__(self, config: MultiModalDBNConfig):
        super().__init__(config)
        self.config = config
        self.tabular_config = config.tabular_config
        if config.use_modality['num']:
            self.num_encoder = MLP(
                input_dim=self.tabular_config.numerical_feat_dim,
                output_dim=512,
                act=self.tabular_config.mlp_act,
                dropout_prob=self.tabular_config.mlp_dropout,
                num_hidden_lyr=3,
                return_layer_outs=False,
                bn=True)
            self.num_dbn = self.create_dbn(n_layer=2, n_visible=512, n_hidden=256, rbm_type='gaussian_relu4deep')
            self.num_bn = nn.BatchNorm1d(256)
        if config.use_modality['cat']:
            self.cat_encoder = MLP(
                input_dim=self.tabular_config.cat_feat_dim,
                output_dim=256,
                act=self.tabular_config.mlp_act,
                dropout_prob=self.tabular_config.mlp_dropout,
                num_hidden_lyr=3,
                return_layer_outs=False,
                bn=True)
            self.cat_dbn = self.create_dbn(n_layer=2, n_visible=256, n_hidden=256, rbm_type='gaussian_relu4deep')
            self.cat_bn = nn.BatchNorm1d(256)
        if config.n_modality > 1:
            self.joint_dbn = self.create_dbn(n_layer=2, n_visible=256 * 2, n_hidden=256, rbm_type='gaussian_relu4deep')

        self.classifier = MLP(256,
                              self.tabular_config.num_labels,
                              num_hidden_lyr=2,
                              dropout_prob=self.tabular_config.mlp_dropout,
                              hidden_channels=[128, 64],
                              bn=True)

        self.no_dbn_training = False

    def forward(self, cat_feats, numerical_feats, labels, attention_mask, input_ids, token_type_ids):
        # params labels, attention_mask, input_ids, token_type_ids are not used to avoid log warning

        if self.config.use_modality['num']:
            num_encoded = self.num_encoder(numerical_feats)
            num_dbn_out = self.fit_or_forward_dbn(self.num_dbn, 'num', num_encoded, epoch=self.config.dbn_train_epoch)

            if self.config.n_modality == 1:
                loss, logits, classifier_layer_outputs = hf_loss_func(num_dbn_out,
                                                                      self.classifier,
                                                                      labels,
                                                                      self.tabular_config.num_labels,
                                                                      None)
                return loss, logits, classifier_layer_outputs

        if self.config.use_modality['cat']:
            cat_encoded = self.cat_encoder(cat_feats)
            cat_dbn_out = self.fit_or_forward_dbn(self.cat_dbn, 'cat', cat_encoded, epoch=self.config.dbn_train_epoch)

        if self.config.n_modality > 1:
            num_dbn_bn_out = self.num_bn(num_dbn_out)
            # make_dot(num_dbn_out, show_attrs=True, ).render("num_dbn_out", directory='./', format="pdf")
            cat_dbn_bn_out = self.num_bn(cat_dbn_out)
            dbn_out_fused = torch.concatenate([num_dbn_bn_out, cat_dbn_bn_out], dim=1)
            # dbn_out_fused = torch.concatenate([num_dbn_out, cat_dbn_out], dim=1)
            joint_dbn_out = self.fit_or_forward_dbn(self.joint_dbn, 'joint', dbn_out_fused,
                                                    epoch=self.config.dbn_train_epoch)

            # ???? I am not sure: because in fit_or_forward_dbn(feature), `feature` is detached from compute graph,
            # hence we need to reforward use `num_encoded` and `cat_encoded` ???
            # num_dbn_out_rf = self.num_dbn(num_encoded)
            # make_dot(num_dbn_out_rf, show_attrs=True, ).render("num_dbn_out_rf", directory='./', format="pdf")
            # cat_dbn_out_rf = self.cat_dbn(cat_encoded)
            # dbn_out_fused_rf = torch.concatenate([num_dbn_out_rf, cat_dbn_out_rf], dim=1)
            # joint_dbn_out_rf = self.joint_dbn(dbn_out_fused_rf)
            loss, logits, classifier_layer_outputs = hf_loss_func(joint_dbn_out,
                                                                  self.classifier,
                                                                  labels,
                                                                  self.tabular_config.num_labels,
                                                                  None)
            return loss, logits, classifier_layer_outputs

    def create_dbn(self, n_layer, n_visible, n_hidden, rbm_type='sigmoid4deep'):
        dbn = CRMMDBN(
            model=[rbm_type for _ in range(n_layer)],
            n_visible=n_visible,
            n_hidden=[n_hidden for _ in range(n_layer)],
            steps=[1 for _ in range(n_layer)],
            learning_rate=[0.01 for _ in range(n_layer)],
            momentum=[0 for _ in range(n_layer)],
            decay=[0 for _ in range(n_layer)],
            temperature=[1 for _ in range(n_layer)],
            use_gpu=self.config.use_gpu,
        )
        return dbn

    def fit_or_forward_dbn(self, dbn, name, feature, epoch):
        if self.training and not self.no_dbn_training:
            # logger.info(f'fit dbn: {name}')
            mse, pl = dbn.fit(feature.detach(), epochs=[epoch for _ in range(dbn.n_layers)])
            hidden_out = dbn(feature)
        else:
            # logger.info(f'forward dbn: {name}')
            hidden_out = dbn(feature)
        return hidden_out

    def stop_dbn_training(self):
        logger.warning('stop_dbn_training!')
        self.no_dbn_training = True
