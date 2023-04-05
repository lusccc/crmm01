from transformers import BertConfig, BertModel

from .bert_text_feat import TextFeatureExtractor
from .emb_cat_feat import CatFeatureExtractor
from .gaussian_gaaussian_rbm import GaussianGaussianRBM
from .joint_feat import JointFeatureExtractor
from .resnet_num_feat import NumFeatureExtractor
import torch.nn.functional as F


class RBMFactory:
    rbm_input_dims = {
        'num': 256,
        'cat': 256,
        'text': 256,
        'joint': 256,
    }
    rbm_output_dims = {
        'num': 256,
        'cat': 256,
        'text': 256,
        'joint': 256,
    }

    def __init__(self, mm_model_config, ):
        self.mm_model_config = mm_model_config
        self.rbms = None

    def get_rbm_output_dim_for_classification(self):
        if len(self.mm_model_config.use_modality) == 1:
            return self.rbm_output_dims[self.mm_model_config.use_modality[0]]
        elif len(self.mm_model_config.use_modality) > 1:
            return self.rbm_output_dims['joint']
        else:
            raise ValueError(f'number of modality {len(self.mm_model_config.use_modality)} not supported')

    def get_rbms(self):
        rbms = {m: self._create_rbm_for(m) for m in self.mm_model_config.use_modality}
        if len(rbms) > 1:
            rbms['joint'] = self._create_rbm_for('joint')
        for modality, rbm in rbms.items():
            rbm.pretrained = self.mm_model_config.pretrained
        self.rbms = rbms
        return rbms

    def _create_rbm_for(self, modality):
        rbm_input_dim = self.rbm_input_dims[modality]
        rbm_output_dim = self.rbm_output_dims[modality]
        if modality == 'num':
            rbm = GaussianGaussianRBM(
                name='num',
                encoder=NumFeatureExtractor(
                    input_dim=self.mm_model_config.num_feat_dim,
                    hidden_dims=[512, 512],  # hidden size in res block
                    output_dim=rbm_input_dim,  # should be equal to visible_units
                    dropout_rate=.2
                ),
                visible_units=rbm_input_dim,
                hidden_units=rbm_output_dim,
                dropout=.2,
            )
        elif modality == 'cat':
            rbm = GaussianGaussianRBM(
                name='cat',
                encoder=CatFeatureExtractor(
                    num_embeddings=self.mm_model_config.n_cat,
                    embedding_dims=self.mm_model_config.cat_emb_dims,
                    hidden_dim=128,
                    output_dim=rbm_input_dim,
                    dropout_prob=.2
                ),
                visible_units=rbm_input_dim,
                hidden_units=rbm_output_dim,
                dropout=.2,
            )
        elif modality == 'text':
            rbm = GaussianGaussianRBM(
                name='text',
                encoder=TextFeatureExtractor(
                    bert_params=self.mm_model_config.bert_params,
                    load_hf_pretrained=not self.mm_model_config.pretrained,
                    output_dim=rbm_input_dim,
                ),
                visible_units=rbm_input_dim,
                hidden_units=rbm_output_dim,
                dropout=.2,
            )
        elif modality == 'joint':
            modality_feat_dims = {m: self.rbm_output_dims[m] for m in self.mm_model_config.use_modality}
            rbm = GaussianGaussianRBM(
                name='joint',
                encoder=JointFeatureExtractor(
                    modality_feat_dims,
                    hidden_dim=256,
                    output_dim=rbm_input_dim,
                ),
                visible_units=rbm_input_dim,
                hidden_units=rbm_output_dim,
                dropout=.2,
            )
        else:
            raise ValueError(f"Invalid modality: {modality}")
        return rbm
