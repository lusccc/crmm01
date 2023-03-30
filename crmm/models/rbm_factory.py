from transformers import BertConfig, BertModel

from .bert_text_feat import TextFeatureExtractor
from .emb_cat_feat import CatFeatureExtractor
from .gaussian_gaaussian_rbm import GaussianGaussianRBM
from .resnet_num_feat import NumFeatureExtractor


class RBMFactory:

    def __init__(self, modalities, mm_model_config, bert_params=None, pretrained=False):
        self.modalities = modalities
        self.mm_model_config = mm_model_config
        self.bert_params = bert_params
        self.pretrained = pretrained

        self.rbms = None

    def get_rbm_output_dim_for_classification(self):
        # for now, each rbm, including joint rbm, its output dim (hidden units) is 256
        return 256

    def get_rbms(self):
        rbms = {m: self._create_rbm_for(m) for m in self.modalities}
        if len(rbms) > 1:
            rbms['joint'] = self._create_joint_rbm_from(bottom_rbms=rbms)
        for modality, rbm in rbms.items():
            rbm.pretrained = self.pretrained
        self.rbms = rbms
        return rbms

    def _create_rbm_for(self, modality):
        if modality == 'num':
            rbm = GaussianGaussianRBM(
                visible_units=256,
                hidden_units=256,
                dropout=.2,
                encoder=NumFeatureExtractor(
                    input_dim=self.mm_model_config.num_feat_dim,
                    hidden_dims=[512, 512],
                    output_dim=256,  # should be equal to visible_units
                    dropout_rate=.1
                ),
            )
        elif modality == 'cat':
            rbm = GaussianGaussianRBM(
                visible_units=256,
                hidden_units=256,
                dropout=.2,
                encoder=CatFeatureExtractor(
                    num_embeddings=self.mm_model_config.n_cat,
                    embedding_dims=self.mm_model_config.cat_emb_dims,
                    hidden_dim=128,
                    output_dim=256,
                    dropout_prob=.2
                ),
            )
        elif modality == 'text':
            rbm = GaussianGaussianRBM(
                visible_units=768,
                hidden_units=256,
                dropout=.2,
                encoder=TextFeatureExtractor(
                    bert=self._get_bert()
                ),
            )
        else:
            raise ValueError(f"Invalid modality: {modality}")

        return rbm

    def _create_joint_rbm_from(self, bottom_rbms):
        total_hidden = sum([rbm.hidden_units for rbm in bottom_rbms.values()])
        joint_rbm = GaussianGaussianRBM(visible_units=total_hidden,
                                        hidden_units=256,
                                        dropout=.2)
        return joint_rbm

    def _get_bert(self):
        if self.pretrained:
            # in fine tune from our pretrained model, we only create bert model with config.
            # the bert part will load with our pretrained MultiModalDBN
            bert_config = BertConfig.from_pretrained(**self.bert_params)
            bert_model = BertModel(config=bert_config)
        else:
            # in pretrain or fine tune from scratch , we need to load the hf pretrained model
            bert_model = BertModel.from_pretrained(**self.bert_params)
        return bert_model


