import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, PreTrainedModel, BertConfig


class TextFeatureExtractor(nn.Module):
    def __init__(self, bert_params, load_hf_pretrained=True, output_dim=768):
        super(TextFeatureExtractor, self).__init__()
        self.bert_params = bert_params
        self.load_hf_pretrained = load_hf_pretrained
        self.output_dim = output_dim

        self.pretrained_model_name_or_path = self.bert_params['pretrained_model_name_or_path']

        if self.load_hf_pretrained:
            # in pretrain or fine tune from scratch , we need to load the hf pretrained model
            self.bert = BertModel.from_pretrained(**self.bert_params)
        else:
            # in fine tune from our pretrained model, we only create bert model with config.
            # the bert part will load with our pretrained MultiModalDBN
            bert_config = BertConfig.from_pretrained(**self.bert_params)
            self.bert = BertModel(config=bert_config)

        self.fc = nn.Linear(self.bert.config.hidden_size, self.output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # pooled_output usually used as features for classification
        pooled_output = outputs[1]
        out = F.gelu(self.fc(pooled_output))
        return out
