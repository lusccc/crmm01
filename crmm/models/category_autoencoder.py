from torch import nn
from torch.nn import MSELoss
from transformers import PretrainedConfig, PreTrainedModel

from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model.layer_utils import MLP


class CategoryAutoencoderConfig(PretrainedConfig):
    model_type = 'cat_ae'

    def __init__(self, emb_dim=64, tabular_config=None, **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.tabular_config = tabular_config

class CategoryAutoencoder(PreTrainedModel):
    def __init__(self, config: CategoryAutoencoderConfig):
        super().__init__(config)
        self.tabular_config = config.tabular_config
        if type(self.tabular_config) is dict:  # when loading from saved model
            self.tabular_config = TabularConfig(**self.tabular_config)
        else:
            self.config.tabular_config = self.tabular_config.__dict__
        self.emb_dim = config.emb_dim
        self.loss_fn = MSELoss()

        self.encoder = MLP(
            self.tabular_config.cat_feat_dim,
            self.emb_dim,
            act=self.tabular_config.mlp_act,
            dropout_prob=self.tabular_config.mlp_dropout,
            num_hidden_lyr=1,
            return_layer_outs=False,
            bn=True)

        self.decoder = MLP(
            self.emb_dim,
            self.tabular_config.cat_feat_dim,
            act=self.tabular_config.mlp_act,
            dropout_prob=self.tabular_config.mlp_dropout,
            num_hidden_lyr=1,
            return_layer_outs=False,
            bn=True)

    def forward(self, cat_feats, numerical_feats, labels, attention_mask, input_ids, token_type_ids):
        # params labels, attention_mask, input_ids, token_type_ids are not used to avoid log warning
        enc_out = self.encoder(cat_feats)
        dec_out = self.decoder(enc_out)
        loss = self.loss_fn(cat_feats, dec_out)
        logits = enc_out
        return loss, logits
