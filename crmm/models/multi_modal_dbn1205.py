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
from transformers import PreTrainedModel, PretrainedConfig, EvalPrediction, BertModel, BertForSequenceClassification

import analysis
from learnergy.models.deep import DBN
from metrics import calc_classification_metrics
from models.crmm_dbn import CrmmDBN
from models.params_exposed_dbn import ParamsExposedDBN
from multimodal_transformers.model.layer_utils import calc_mlp_dims, MLP, hf_loss_func
import numpy as np
from transformers.utils import logging
from utils import utils

logger = logging.get_logger('transformers')

# TODO only use single DBN to fuse
class MultiModalConfig(PretrainedConfig):
    model_type = 'multi_modal_dbn'

    modality_list = ('num', 'cat', 'text')

    def __init__(self,
                 tabular_config=None,
                 bert_config=None,
                 use_modality=modality_list,
                 dbn_train_epoch=2,
                 use_gpu=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.tabular_config = tabular_config
        self.bert_config = bert_config
        self.n_modality = len(use_modality)
        self.use_modality = {
            'num': True if 'num' in use_modality else False,
            'cat': True if 'cat' in use_modality else False,
            'text': True if 'text' in use_modality else False,
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


class MultiModalBertDBN(PreTrainedModel):
    def __init__(self, config: MultiModalConfig):
        super().__init__(config)
        self.config = config
        self.tabular_config = self.config.tabular_config
        """MAIN try: not use DBN for each modality"""
        if self.config.use_modality['num']:
            self.num_encoder = MLP(
                input_dim=self.tabular_config.numerical_feat_dim,
                output_dim=512,
                act=self.tabular_config.mlp_act,
                dropout_prob=self.tabular_config.mlp_dropout,
                num_hidden_lyr=3,
                return_layer_outs=False,
                bn=True)
        """MAIN try: not use DBN for each modality"""
        if self.config.use_modality['cat']:
            self.cat_encoder = MLP(
                input_dim=self.tabular_config.cat_feat_dim,
                output_dim=256,
                act=self.tabular_config.mlp_act,
                dropout_prob=self.tabular_config.mlp_dropout,
                num_hidden_lyr=3,
                return_layer_outs=False,
                bn=True)

        if self.config.use_modality['text']:
            # to make compatible with from_pretrained `bert-base-uncased`, it's named as `bert`
            # see `transformers.modeling_utils.PreTrainedModel._load_pretrained_model` for detail
            self.bert = BertModel(self.config.bert_config)

        if self.config.n_modality > 1:
            self.joint_dbn = self.create_dbn(n_layer=2, n_visible=512+256 + 768, n_hidden=256,
                                             rbm_type='gaussian_relu4deep')

        self.classifier = MLP(256,
                              self.tabular_config.num_labels,
                              num_hidden_lyr=2,
                              dropout_prob=self.tabular_config.mlp_dropout,
                              hidden_channels=[128, 64],
                              bn=True)

        self.no_dbn_training = False

    def _init_weights(self, module):
        pass

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                class_weights=None,
                output_attentions=None,
                output_hidden_states=None,
                cat_feats=None,
                numerical_feats=None):
        # params labels, attention_mask, input_ids, token_type_ids are not used to avoid log warning

        if self.config.use_modality['num']:
            num_encoded = self.num_encoder(numerical_feats)
        if self.config.use_modality['cat']:
            cat_encoded = self.cat_encoder(cat_feats)
        if self.config.use_modality['text']:
            bert_out = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            bert_out_pooled = bert_out[1]

        if self.config.n_modality > 1:
            out_fused = torch.concatenate([num_encoded, cat_encoded, bert_out_pooled], dim=1)

            # dbn_out_fused = torch.concatenate([num_dbn_bn_out, cat_dbn_bn_out], dim=1)
            # dbn_out_fused = torch.concatenate([num_dbn_out, cat_dbn_out], dim=1)

            joint_dbn_out = self.fit_or_forward_dbn(self.joint_dbn, 'joint', out_fused,
                                                    epoch=self.config.dbn_train_epoch)
            # make_dot(joint_dbn_out, show_attrs=True, ).render("joint_dbn_out", directory='./', format="pdf")


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
        dbn = CrmmDBN(
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