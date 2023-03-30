import json
from typing import Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, BertModel, BertConfig
from transformers.utils import logging

from .layer_utils import get_classifier, hf_loss_func
from .rbm_factory import RBMFactory

logger = logging.get_logger('transformers')


class MultiModalModelConfig(PretrainedConfig):
    def __init__(self,
                 n_labels=0,
                 num_feat_dim=0,
                 cat_feat_dim=0,
                 n_cat=0,
                 cat_emb_dims=0,
                 use_modality=None,
                 bert_params=None,
                 pretrained=False,
                 **kwargs):
        super().__init__(**kwargs)

        # unique label count
        self.n_labels = n_labels
        # numerical feature dimension after num_transformer in dataset.crmm_data.MultimodalData.transform_features
        self.num_feat_dim = num_feat_dim
        # category feature dimension after categorized in dataset.crmm_data.MultimodalData.transform_features
        self.cat_feat_dim = cat_feat_dim
        # category features count
        self.n_cat = n_cat
        # category feature embedding dimension, used in Embedding layer in models.emb_cat_feat.CatFeatureExtractor
        self.cat_emb_dims = cat_emb_dims

        self.use_modality = use_modality
        self.bert_params = bert_params
        self.pretrained = pretrained


class MultiModalDBNPretrainedModel(PreTrainedModel):
    config_class = MultiModalModelConfig
    base_model_prefix = "mmdbn"


class MultiModalDBN(MultiModalDBNPretrainedModel):
    pretrained: bool = False

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.use_modality = self.config.use_modality
        self.n_modality = len(self.use_modality)
        self.n_labels = self.config.n_labels
        MultiModalDBN.pretrained = self.config.pretrained

        self.pretrain_target = None

        self.rbm_factory = RBMFactory(modalities=self.use_modality,
                                      mm_model_config=self.config,
                                      bert_params=self.config.bert_params,
                                      pretrained=self.pretrained)
        self.rbms = self.rbm_factory.get_rbms()

        # it is necessary to register rbm as module
        self._modules.update(self.rbms)

    def set_pretrain_target(self, pretrain_target):
        logger.info(f'set pretrain_target: {pretrain_target}')
        self.pretrain_target = pretrain_target

    def pretrain_step(self, inputs):
        if self.pretrain_target == 'modality_rbm':
            loss = 0
            for modality in self.use_modality:
                rbm_loss = self.rbms[modality](inputs[modality])
                loss += rbm_loss
        elif self.pretrain_target == 'joint_rbm':
            hidden_outputs = []
            for modality in self.use_modality:
                hidden_output = self.rbms[modality].extra_features_step(inputs[modality])
                # should be DETACHED then passed to joint_rbm, according to the RBM training method
                hidden_outputs.append(hidden_output.detach())
            cat_hidden_outputs = torch.cat(hidden_outputs, dim=1)
            loss = self.rbms['joint'](cat_hidden_outputs)
        else:
            loss = None
            raise ValueError(f'pretrain_target {self.pretrain_target} not supported')

        return {'loss': loss}

    def extra_features_step(self, inputs):
        if self.n_modality == 1:
            modality = self.use_modality[0]
            features = self.rbms[modality](inputs[modality])
        elif self.n_modality > 1:
            hidden_outputs = []
            for modality in self.use_modality:
                hidden_output = self.rbms[modality](inputs[modality])
                # don't need to detach, since it is not pretraining
                hidden_outputs.append(hidden_output)
            cat_hidden_outputs = torch.cat(hidden_outputs, dim=1)
            features = self.rbms['joint'](cat_hidden_outputs)
        else:
            features = None
            raise ValueError(f'number of modality {self.n_modality} not supported')
        return features

    def forward(self, return_loss=True, **inputs, ):
        # should return_loss=True, or cause error in eval
        # `return_loss=` will be checked in transformers.utils.generic.can_return_loss

        # inputs: {'labels': ...,
        #         'text': {'input_ids',:..., 'attention_mask':...},
        #         'num': ...,
        #         'cat': ...}
        return self.extra_features_step(inputs) if self.pretrained else self.pretrain_step(inputs)

    def _init_weights(self, module):
        pass


class MultiModalForClassification(MultiModalDBNPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mmdbn = MultiModalDBN(self.config)
        self.n_labels = self.config.n_labels

        self.classifier = get_classifier(input_dim=self.mmdbn.rbm_factory.get_rbm_output_dim_for_classification(),
                                         n_class=self.n_labels)

    def forward(self, labels, return_loss=True, **kwargs):
        # _ = kwargs.pop('labels')
        inputs = kwargs
        classification_features = self.mmdbn(**inputs)
        loss, logits, classifier_layer_outputs = hf_loss_func(classification_features,
                                                              self.classifier,
                                                              labels,
                                                              self.n_labels,
                                                              None)
        # https://stackoverflow.com/questions/61414065/pytorch-weight-in-cross-entropy-loss
        # see 8class_weight.ipynb
        # class_weights=torch.Tensor([3, 0.830040, 0.513447, 0.695364,
        #                1.166667, 4.565217]).cuda())
        return loss, logits, classifier_layer_outputs

    def _init_weights(self, module):
        pass
