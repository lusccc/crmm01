from collections.abc import Mapping
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import BertTokenizerFast, DefaultDataCollator
from transformers.data.data_collator import torch_default_data_collator


class TorchTabularTextDataset(TorchDataset):
    """
    :obj:`TorchDataset` wrapper for text dataset with categorical features
    and numerical features

    Parameters:
        encodings (:class:`transformers.BatchEncoding`):
            The output from encode_plus() and batch_encode() methods (tokens, attention_masks, etc) of
            a transformers.PreTrainedTokenizer
        categorical_feats (:class:`numpy.ndarray`, of shape :obj:`(n_examples, categorical feat dim)`, `optional`, defaults to :obj:`None`):
            An array containing the preprocessed categorical features
        numerical_feats (:class:`numpy.ndarray`, of shape :obj:`(n_examples, numerical feat dim)`, `optional`, defaults to :obj:`None`):
            An array containing the preprocessed numerical features
        labels (:class: list` or `numpy.ndarray`, `optional`, defaults to :obj:`None`):
            The labels of the training examples
        class_weights (:class:`numpy.ndarray`, of shape (n_classes),  `optional`, defaults to :obj:`None`):
            Class weights used for cross entropy loss for classification
        df (:class:`pandas.DataFrame`, `optional`, defaults to :obj:`None`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            TabularConfig instance specifying the configs for TabularFeatCombiner

    """

    def __init__(self,
                 texts_list,
                 categorical_feats,
                 numerical_feats,
                 labels=None,
                 df=None,
                 label_list=None,
                 class_weights=None
                 ):
        self.df = df
        self.texts_list = texts_list
        self.cat_feats = categorical_feats
        self.numerical_feats = numerical_feats
        self.labels = labels
        self.class_weights = class_weights
        self.label_list = label_list if label_list is not None else [i for i in range(len(np.unique(labels)))]

    def __getitem__(self, idx):
        # note texts_list wrapped with list
        # enc = self.tokenizer([self.texts_list[idx]], padding=True, truncation=True,
        #                      max_length=self.max_token_length, return_tensors="pt")
        # item = {k: torch.tensor(v)
        #         for k, v in enc.items()}
        untokenized = self.texts_list[idx]
        item = {'labels': torch.tensor(self.labels[idx]) if self.labels is not None else None,
                'cat_feats': torch.tensor(self.cat_feats[idx]).float() \
                    if self.cat_feats is not None else torch.zeros(0),
                'numerical_feats': torch.tensor(self.numerical_feats[idx]).float() \
                    if self.numerical_feats is not None else torch.zeros(0)}
        # return item
        return item, untokenized

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        """returns the label names for classification"""
        return self.label_list


class TabularTextCollator:

    def __init__(self, tokenizer, max_token_length=None) -> None:
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __call__(self, features):
        first = features[0]
        has_untokenized = True if isinstance(first, tuple) else False
        if has_untokenized:
            txts = [f[1] for f in features]
            tokenized = self.tokenizer(txts, padding=True, truncation=True,
                                       max_length=self.max_token_length, return_tensors="pt")

            # reorganize to dict
            features = [f[0] for f in features]
            t_keys = tokenized.keys()
            for i, f in enumerate(features):
                for k in t_keys:
                    f[k] = tokenized[k][i]

        features = torch_default_data_collator(features)
        return features
