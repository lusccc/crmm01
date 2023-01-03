import json
import os.path
import traceback
from typing import Tuple, Optional, List

import numpy
import torch
from torchviz import make_dot

from learnergy.core import Dataset
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import PreTrainedModel, PretrainedConfig, EvalPrediction, BertModel, BertForSequenceClassification, \
    BertConfig

import analysis
from learnergy.models.deep import DBN
from metrics import calc_classification_metrics
from models.crmm_dbn import CrmmDBN
from models.crmm_gaussian_rbm import CrmmGaussianReluRBM4deep
from models.params_exposed_dbn import ParamsExposedDBN
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model.layer_utils import calc_mlp_dims, MLP, hf_loss_func
import numpy as np
from transformers.utils import logging
from utils import utils
from utils.utils import dict_to_obj

logger = logging.get_logger('transformers')


class MultiModalConfig(PretrainedConfig):
    model_type = 'multi_modal_dbn'

    modality_list = ('num', 'cat', 'text')

    def __init__(self,
                 tabular_config=None,
                 bert_config=None,
                 use_modality=modality_list,
                 dbn_train_epoch=2,
                 pretrain=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.tabular_config = tabular_config
        self.bert_config = bert_config
        self.n_modality = len(use_modality)
        self.use_modality = use_modality
        if not isinstance(self.use_modality, dict): # because load pretrainedConfig result is a dict
            self.use_modality = {
                'num': True if 'num' in use_modality else False,
                'cat': True if 'cat' in use_modality else False,
                'text': True if 'text' in use_modality else False,
            }
        self.dbn_train_epoch = dbn_train_epoch
        self.rbm_train_epoch = dbn_train_epoch
        self.pretrain = pretrain

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
        self.bert_config = self.config.bert_config
        if isinstance(self.tabular_config, dict):
            self.tabular_config = dict_to_obj(TabularConfig, self.tabular_config)
        if isinstance(self.bert_config, dict):
            self.bert_config = dict_to_obj(BertConfig, self.bert_config)

        self.rbms = {}
        if self.config.use_modality['num']:
            self.num_encoder = MLP(
                input_dim=self.tabular_config.numerical_feat_dim,
                output_dim=256,  # should equal to n_hidden below
                act=self.tabular_config.mlp_act,
                dropout_prob=self.tabular_config.mlp_dropout,
                num_hidden_lyr=3,
                return_layer_outs=False,
                bn=True)
            """main try: modify rbm, visible_layer!!!"""
            self.num_rbm = self.create_rbm(n_visible=self.tabular_config.numerical_feat_dim,
                                           n_hidden=256,
                                           visible_layer=self.num_encoder,
                                           input_norm=True,
                                           visible_out_norm=True)
            self.rbms['num'] = self.num_rbm
            classifier_input_dim = 256

        if self.config.use_modality['cat']:
            self.cat_encoder = MLP(
                input_dim=self.tabular_config.cat_feat_dim,
                output_dim=256,
                act=self.tabular_config.mlp_act,
                dropout_prob=self.tabular_config.mlp_dropout,
                num_hidden_lyr=3,
                return_layer_outs=False,
                bn=True)
            self.cat_rbm = self.create_rbm(n_visible=self.tabular_config.cat_feat_dim,
                                           n_hidden=256,
                                           visible_layer=self.cat_encoder,
                                           input_norm=True,
                                           visible_out_norm=True)
            self.rbms['cat'] = self.cat_rbm
            self.classifier_input_dim = 256

        if self.config.use_modality['text']:
            # to make compatible with from_pretrained `bert-base-uncased`, it's named as `bert`
            # see `transformers.modeling_utils.PreTrainedModel._load_pretrained_model` for detail
            self.bert = CompatibleBert(self.bert_config)
            """note the visible layer is `self.bert.encoder`"""
            # note: we don't normalize, because bert model already layernorm!
            self.text_rbm = self.create_rbm(n_visible=768,
                                            n_hidden=768,
                                            visible_layer=self.bert,
                                            input_norm=False,
                                            visible_out_norm=False)
            self.rbms['text'] = self.text_rbm
            self.classifier_input_dim = 768

        if self.config.n_modality > 1:
            # TODO joint dbn change to joint rbm!!
            """main change: joint_dbn is changed to joint_rbm !"""
            total_hidden = 0
            for n, r in self.rbms.items():
                total_hidden += r.n_hidden
            # previous rbm hidden is next rbm visible
            self.joint_rbm = self.create_rbm(n_visible=total_hidden, n_hidden=512, input_norm=False,
                                             visible_out_norm=False)
            self.classifier_input_dim = 512

        if not self.config.pretrain:
            self.classifier = MLP(self.classifier_input_dim,
                                  self.tabular_config.num_labels,
                                  num_hidden_lyr=3,
                                  dropout_prob=self.tabular_config.mlp_dropout,
                                  hidden_channels=[256, 128, 64],
                                  bn=True)

        self.no_dbn_rbm_training = not self.config.pretrain

    def _init_weights(self, module):
        pass

    def stop_dbn_rbm_training(self):
        self.no_dbn_rbm_training = True

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
        """"""
        """main try: use modified rbm!"""
        """
            to pretrain (fit) rbm, dbm,  only need to train 1 epoch of this module (MultiModalBertDBN),
            in forward, it will fit rbm or for epoch of dbn self.config.rbm_train_epoch
        """
        # print('\n  ------begin MultiModalBertDBN forward------')
        rbm_outs = []
        if self.config.use_modality['num']:
            num_rbm_out = self.fit_or_forward_rbm(self.num_rbm, 'num', numerical_feats, self.config.rbm_train_epoch)
            rbm_outs.append(num_rbm_out)

        if self.config.use_modality['cat']:
            cat_rbm_out = self.fit_or_forward_rbm(self.cat_rbm, 'cat', cat_feats, self.config.rbm_train_epoch)
            rbm_outs.append(cat_rbm_out)

        if self.config.use_modality['text']:
            if self.config.pretrain:
                first_input = self.bert.get_first_input(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                # print('\n  text_rbm fit_or_forward_rbm')
                # TODO may be problem!
                """note: for fitting text rbm, due to its CD training method,
                some input above (the input that bert specific) may missing"""
                text_rbm_out = self.fit_or_forward_rbm(self.text_rbm, 'text', first_input, self.config.rbm_train_epoch)
                text_rbm_out = self.bert.pooler(text_rbm_out)
            else:
                """ for fine tuning, the input should be complete, hence we do below as `BertModel` """
                # we don't pass `feature` to models.multi_modal_dbn.CompatibleBert.forward !
                # below will return pooler_output
                text_rbm_out = self.bert(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         position_ids=position_ids,
                                         head_mask=head_mask,
                                         inputs_embeds=inputs_embeds,
                                         output_attentions=output_attentions,
                                         output_hidden_states=output_hidden_states, )

            rbm_outs.append(text_rbm_out)

        if self.config.n_modality > 1:
            if self.config.pretrain:
                # note detach in pretrain dbn !!!!! according to the CD training algorithm. DBN (rbm) train two layers once!
                out_fused = torch.concatenate([o.detach() for o in rbm_outs], dim=1)
            else:
                out_fused = torch.concatenate(rbm_outs, dim=1)

            # print('  fit joint_rbm')
            joint_rbm_out = self.fit_or_forward_rbm(self.joint_rbm, 'joint', out_fused, self.config.dbn_train_epoch)
            rbm_outs.append(joint_rbm_out)

        final_dbn_out = rbm_outs[-1]  # for single modality, -1 also means the first rbm out
        if not self.config.pretrain:
            loss, logits, classifier_layer_outputs = hf_loss_func(final_dbn_out,
                                                                  self.classifier,
                                                                  labels,
                                                                  self.tabular_config.num_labels,
                                                                  None)
            return loss, logits, classifier_layer_outputs

    def create_rbm(self, n_visible, n_hidden, visible_layer=None, input_norm=True, visible_out_norm=False):
        rbm = CrmmGaussianReluRBM4deep(n_visible, n_hidden, visible_layer=visible_layer,
                                       visible_out_normalize=visible_out_norm, input_normalize=input_norm)
        return rbm

    def fit_or_forward_rbm(self, rbm, name, feature, epoch):
        # print(f'fit_or_forward_rbm {name}')
        # TODO input_normalize?learnergy.models.gaussian.gaussian_rbm.GaussianRBM.forward
        if not self.no_dbn_rbm_training:
            # print('do fit!')
            mse, pl = rbm.fit(feature, epoch, name=name)
        hidden_out = rbm(feature)
        return hidden_out


class CompatibleBert(BertModel):
    def forward(self, feature=None, **kwargs):

        if feature is not None:
            # return sequence_output (last_hidden_state)
            return self.encoder(feature)[0]
        else:
            # return pooled_output
            return super().forward(**kwargs)[1]

    def get_first_input(self,
                        input_ids: Optional[torch.Tensor] = None,
                        attention_mask: Optional[torch.Tensor] = None,
                        token_type_ids: Optional[torch.Tensor] = None,
                        position_ids: Optional[torch.Tensor] = None,
                        head_mask: Optional[torch.Tensor] = None,
                        inputs_embeds: Optional[torch.Tensor] = None,
                        encoder_hidden_states: Optional[torch.Tensor] = None,
                        encoder_attention_mask: Optional[torch.Tensor] = None,
                        past_key_values: Optional[List[torch.FloatTensor]] = None,
                        use_cache: Optional[bool] = None,
                        output_attentions: Optional[bool] = None,
                        output_hidden_states: Optional[bool] = None,
                        return_dict: Optional[bool] = None,
                        ):

        """
        copied from transformers.models.bert.modeling_bert.BertModel.forward
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        return embedding_output
