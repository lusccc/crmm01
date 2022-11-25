import json
from abc import ABC

from torch import nn
from torch.nn import MSELoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model.layer_utils import MLP


class NumericalAutoencoderConfig(PretrainedConfig):
    model_type = 'num_ae'

    def __init__(self, emb_dim=64, tabular_config=None, **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.tabular_config = tabular_config


class NumericalAutoencoder(PreTrainedModel):
    def __init__(self, config: NumericalAutoencoderConfig):
        super().__init__(config)
        self.tabular_config = config.tabular_config
        if type(self.tabular_config) is dict:  # when loading from saved model
            self.tabular_config = TabularConfig(**self.tabular_config)
        else:
            self.config.tabular_config = self.tabular_config.__dict__
        self.emb_dim = config.emb_dim
        self.loss_fn = MSELoss()

        if self.tabular_config.numerical_bn and self.tabular_config.numerical_feat_dim > 0:
            self.num_bn = nn.BatchNorm1d(self.tabular_config.numerical_feat_dim)
        else:
            self.num_bn = None

        self.encoder = MLP(
            self.tabular_config.numerical_feat_dim,
            self.emb_dim,
            act=self.tabular_config.mlp_act,
            dropout_prob=self.tabular_config.mlp_dropout,
            num_hidden_lyr=1,
            return_layer_outs=False,
            bn=True)

        self.decoder = MLP(
            self.emb_dim,
            self.tabular_config.numerical_feat_dim,
            act=self.tabular_config.mlp_act,
            dropout_prob=self.tabular_config.mlp_dropout,
            num_hidden_lyr=1,
            return_layer_outs=False,
            bn=True)

    def forward(self, cat_feats, numerical_feats, labels, attention_mask, input_ids, token_type_ids):
        # params labels, attention_mask, input_ids, token_type_ids are not used to avoid log warning
        if self.tabular_config.numerical_bn and self.tabular_config.numerical_feat_dim != 0:
            numerical_feats = self.num_bn(numerical_feats)
        enc_out = self.encoder(numerical_feats)
        dec_out = self.decoder(enc_out)
        loss = self.loss_fn(numerical_feats, dec_out)
        logits = enc_out
        return loss, logits
