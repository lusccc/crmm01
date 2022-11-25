import torch
from learnergy.core import Dataset
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig

from models.params_exposed_dbn import ParamsExposedDBN
from multimodal_transformers.model.layer_utils import calc_mlp_dims, MLP, hf_loss_func


class MultiModalDBNConfig(PretrainedConfig):
    model_type = 'multi_modal_dbn'

    def __init__(self, dbn_input_dim, dbn_output_dim, dbn_train_epoch, **kwargs):
        super().__init__(**kwargs)
        self.dbn_input_dim = dbn_input_dim
        self.dbn_output_dim = dbn_output_dim
        self.dbn_train_epoch = dbn_train_epoch


class MultiModalDBN(PreTrainedModel):
    def __init__(self, tabular_config, dbn_input_dim, dbn_output_dim, dbn_train_epoch):
        super().__init__()
        self.combine_feat_method = tabular_config.combine_feat_method
        self.cat_feat_dim = tabular_config.cat_feat_dim
        self.numerical_feat_dim = tabular_config.numerical_feat_dim
        self.num_labels = tabular_config.num_labels
        self.numerical_bn = tabular_config.numerical_bn
        self.mlp_act = tabular_config.mlp_act
        self.mlp_dropout = tabular_config.mlp_dropout
        self.mlp_division = tabular_config.mlp_division
        self.tabular_config = tabular_config

        self.dbn_input_dim = dbn_input_dim
        self.output_dim_cat = self.dbn_input_dim
        self.output_dim_num = self.dbn_input_dim
        self.dbn_train_epoch = dbn_train_epoch
        self.dbn_output_dim = dbn_output_dim

        # use_gpu = True
        use_gpu = False
        # self.numerical_feat_dbn = DBN(
        self.numerical_feat_dbn = ParamsExposedDBN(
            model="sigmoid",
            n_visible=self.dbn_input_dim,
            n_hidden=(128, 256, 128),
            steps=(1, 1, 1),
            learning_rate=(0.1, 0.1, 0.1),
            momentum=(0, 0, 0),
            decay=(0, 0, 0),
            temperature=(1, 1, 1),
            use_gpu=use_gpu,
        )

        # self.cat_feat_dbn = DBN(
        self.cat_feat_dbn = ParamsExposedDBN(
            model="sigmoid",
            n_visible=self.dbn_input_dim,
            n_hidden=(128, 256, 128),
            steps=(1, 1, 1),
            learning_rate=(0.1, 0.1, 0.1),
            momentum=(0, 0, 0),
            decay=(0, 0, 0),
            temperature=(1, 1, 1),
            use_gpu=use_gpu,
        )

        # self.joint_dbn = DBN(
        self.joint_dbn = ParamsExposedDBN(
            model="sigmoid",
            n_visible=self.dbn_input_dim,
            n_hidden=(128, 128),
            steps=(1, 1),
            learning_rate=(0.1, 0.1),
            momentum=(0, 0),
            decay=(0, 0),
            temperature=(1, 1),
            use_gpu=use_gpu,
        )

        if self.numerical_bn and self.numerical_feat_dim > 0:
            self.num_bn = nn.BatchNorm1d(self.numerical_feat_dim)
        else:
            self.num_bn = None

        if self.combine_feat_method == 'individual_mlps_on_cat_and_numerical_feats_then_concat':
            if self.cat_feat_dim > 0:
                dims = calc_mlp_dims(
                    self.cat_feat_dim,
                    self.mlp_division,
                    self.output_dim_cat)
                self.cat_mlp = MLP(
                    self.cat_feat_dim,
                    self.output_dim_cat,
                    act=self.mlp_act,
                    num_hidden_lyr=len(dims),
                    dropout_prob=self.mlp_dropout,
                    hidden_channels=dims,
                    return_layer_outs=False,
                    bn=True)

            if self.numerical_feat_dim > 0:
                self.num_mlp = MLP(
                    self.numerical_feat_dim,
                    self.output_dim_num,
                    act=self.mlp_act,
                    dropout_prob=self.mlp_dropout,
                    num_hidden_lyr=1,
                    return_layer_outs=False,
                    bn=True)
            self.final_out_dim = self.dbn_output_dim

            if self.tabular_config.use_simple_classifier:
                self.tabular_classifier = nn.Linear(self.final_out_dim,
                                                    self.tabular_config.num_labels)
            else:
                dims = calc_mlp_dims(self.final_out_dim,
                                     division=self.tabular_config.mlp_division,
                                     output_dim=self.tabular_config.num_labels)
                self.tabular_classifier = MLP(self.final_out_dim,
                                              self.tabular_config.num_labels,
                                              num_hidden_lyr=len(dims),
                                              dropout_prob=self.tabular_config.mlp_dropout,
                                              hidden_channels=dims,
                                              bn=True)

    def forward(self, cat_feats, numerical_feats, labels):
        if self.numerical_bn and self.numerical_feat_dim != 0:
            numerical_feats = self.num_bn(numerical_feats)
        if self.combine_feat_method == 'individual_mlps_on_cat_and_numerical_feats_then_concat':
            if cat_feats.shape[1] != 0:
                cat_feats = self.cat_mlp(cat_feats)
                cat_feats = F.softplus(cat_feats)
            if numerical_feats.shape[1] != 0:
                numerical_feats = self.num_mlp(numerical_feats)
                numerical_feats = F.softplus(numerical_feats)
            # train each dbn
            if self.training:
                self.numerical_feat_dbn = self.fit_dbn(self.numerical_feat_dbn,
                                                       numerical_feats,
                                                       labels,
                                                       [self.dbn_train_epoch, self.dbn_train_epoch,
                                                        self.dbn_train_epoch])
                # self.cat_feat_dbn = self.fit_dbn(self.cat_feat_dbn,
                #                                  cat_feats,
                #                                  labels,
                #                                  [self.dbn_train_epoch, self.dbn_train_epoch, self.dbn_train_epoch])
            # using trained dbn to get output
            numerical_feat_dbn_output = self.numerical_feat_dbn(numerical_feats)
            # cat_feat_dbn_output = self.cat_feat_dbn(cat_feats)
            # dbn_out_feature_added = torch.add(
            #     numerical_feat_dbn_output,
            #     cat_feat_dbn_output,
            # )

            # if self.training:
            #     self.joint_dbn = self.fit_dbn(self.joint_dbn, dbn_out_feature_added, labels,
            #                                   [self.dbn_train_epoch, self.dbn_train_epoch])
            #
            # dbn_feature_fused = self.joint_dbn(dbn_out_feature_added)
            # assert False

            loss, logits, classifier_layer_outputs = hf_loss_func(numerical_feat_dbn_output,
            # loss, logits, classifier_layer_outputs = hf_loss_func(numerical_feats,
                                                                  self.tabular_classifier,
                                                                  labels,
                                                                  self.num_labels, None)
        return loss, logits, classifier_layer_outputs

    def fit_dbn(self, dbn, feature, labels, epoch):
        feature_d = feature.clone().cpu().detach()
        feat_dataset = Dataset(feature_d,
                               targets=labels.clone().cpu(),
                               transform=lambda x: torch.from_numpy(x))  # note targets=feature is just placeholder
        dbn.fit(feat_dataset, batch_size=256, epochs=epoch)
        return dbn
