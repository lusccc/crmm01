from transformers import BertConfig, BertModel

from .bert_text_feat import TextFeatureExtractor
from .emb_cat_feat import CatFeatureExtractor
from .gaussian_gaaussian_rbm import GaussianGaussianRBM
from .joint_feat import JointFeatureExtractor
from .resnet_num_feat import NumFeatureExtractor
import torch.nn.functional as F


class RBMFactory:

    def __init__(self, mm_model_config, ):
        self.mm_model_config = mm_model_config
        self.rbms = None

    def get_rbm_output_dim_for_classification(self):
        if len(self.mm_model_config.use_modality) == 1:
            return self.rbms[self.mm_model_config.use_modality[0]].encoder.get_output_dim()
        elif len(self.mm_model_config.use_modality) > 1:
            return self.rbms['joint'].encoder.get_output_dim()
        else:
            raise ValueError(f'number of modality {len(self.mm_model_config.use_modality)} not supported')

    def get_rbms(self):
        rbms = {m: self._create_rbm_for(m, dropout=.1) for m in self.mm_model_config.use_modality}
        if len(rbms) > 1:
            rbms['joint'] = self._create_joint_rbm(rbms, dropout=.1)
        for modality, rbm in rbms.items():
            rbm.pretrained = self.mm_model_config.pretrained
        self.rbms = rbms
        return rbms

    def _create_rbm_for(self, modality,dropout=.3):
        if modality == 'num':
            feature_extractor = NumFeatureExtractor(
                input_dim=self.mm_model_config.num_feat_dim,
                hidden_dims=[512, 512] if self.mm_model_config.small_params else [512, 512, 512],
                dropout=dropout
            )
            rbm = GaussianGaussianRBM(
                name='num',
                encoder=feature_extractor,
                visible_units=feature_extractor.get_output_dim(),
                hidden_units=feature_extractor.get_output_dim(),
                dropout=dropout
            )
        elif modality == 'cat':
            feature_extractor = CatFeatureExtractor(
                num_embeddings=self.mm_model_config.nunique_cat_nums,
                embedding_dims=self.mm_model_config.cat_emb_dims,
                hidden_dim=max(self.mm_model_config.cat_emb_dims),
                dropout=dropout,
                small_params=self.mm_model_config.small_params
            )
            rbm = GaussianGaussianRBM(
                name='cat',
                encoder=feature_extractor,
                visible_units=feature_extractor.get_output_dim(),
                hidden_units=feature_extractor.get_output_dim(),
                dropout=dropout
            )
        elif modality == 'text':
            feature_extractor = TextFeatureExtractor(
                bert_params=self.mm_model_config.bert_args,
                load_hf_pretrained=self.mm_model_config.use_hf_pretrained_bert,
                freeze_bert_params=self.mm_model_config.freeze_bert_params
            )
            rbm = GaussianGaussianRBM(
                name='text',
                encoder=feature_extractor,
                visible_units=feature_extractor.get_output_dim(),
                hidden_units=feature_extractor.get_output_dim(),
                dropout=dropout
            )
        else:
            raise ValueError(f"Invalid modality: {modality}")
        return rbm

    def _create_joint_rbm(self, bottom_rbms, dropout=.3):
        modality_feat_dims = {m: bottom_rbms[m].encoder.get_output_dim() for m in self.mm_model_config.use_modality}
        feature_extractor = JointFeatureExtractor(
            modality_feat_dims=modality_feat_dims,
            hidden_dims=[512, 512],
            dropout=dropout,
            modality_fusion_method=self.mm_model_config.modality_fusion_method
        )
        joint_rbm = GaussianGaussianRBM(
            name='joint',
            encoder=feature_extractor,
            visible_units=feature_extractor.get_output_dim(),
            hidden_units=feature_extractor.get_output_dim(),
            dropout=dropout
        )
        return joint_rbm
