import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, PreTrainedModel


class TextFeatureExtractor(nn.Module):
    def __init__(self, bert):
        super(TextFeatureExtractor, self).__init__()
        self.bert = bert

    def forward(self, input_ids,attention_mask ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # pooled_output usually used as features for classification
        pooled_output = outputs[1]
        return pooled_output

