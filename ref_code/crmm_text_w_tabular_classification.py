#!/usr/bin/env python
# coding: utf-8
# ## All other imports are here:

# In[9]:


from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed
)
from transformers.training_args import TrainingArguments

from src.multimodal_transformers import load_data_from_folder
from src.multimodal_transformers import TabularConfig
from src.multimodal_transformers import AutoModelWithTabular
import logzero
from logzero import logger

logging.basicConfig(level=logging.INFO)
os.environ['COMET_MODE'] = 'DISABLED'

logzero.logfile('../output1019.log', backupCount=3)

# #### Let us take a look at what the dataset looks like

# In[10]:


# data_df = pd.read_csv('corporate_rating_with_cik.csv')
data_df = pd.read_csv('../data/cr_sec_ori/corporate_rating_with_cik_and_sec_merged_text.csv')
data_df = data_df.replace(
    {'Rating': {'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3, 'BB': 4, 'B': 5, 'CCC': 6, 'CC': 7, 'C': 8, 'D': 9}}
)
data_df.head(5)
data_labels = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC']

# In[11]:



# In this demonstration, we split our data into 8:1:1 training splits. We also save our splits to `train.csv`, `val.csv`, and `test.csv` as this is the format our dataloader requires.
# 

# In[13]:


train_df, val_df, test_df = np.split(data_df.sample(frac=1), [int(.8 * len(data_df)), int(.9 * len(data_df))])
print('Num examples train-val-test')
print(len(train_df), len(val_df), len(test_df))
train_df.to_csv('train.csv')
val_df.to_csv('val.csv')
test_df.to_csv('test.csv')


# ## We then our Experiment Parameters
# We use Data Classes to hold each of our arguments for the model, data, and training. 

# In[14]:


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class MultimodalDataTrainingArguments:
    """
    Arguments pertaining to how we combine tabular features
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_path: str = field(metadata={
        'help': 'the path to the csv file containing the dataset'
    })
    column_info_path: str = field(
        default=None,
        metadata={
            'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'
        })

    column_info: dict = field(
        default=None,
        metadata={
            'help': 'a dict referencing the text, categorical, numerical, and label columns'
                    'its keys are text_cols, num_cols, cat_cols, and label_col'
        })

    categorical_encode_type: str = field(default='ohe',
                                         metadata={
                                             'help': 'sklearn encoder to use for categorical data',
                                             'choices': ['ohe', 'binary', 'label', 'none']
                                         })
    numerical_transformer_method: str = field(default='yeo_johnson',
                                              metadata={
                                                  'help': 'sklearn numerical transformer to preprocess numerical data',
                                                  'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']
                                              })
    task: str = field(default="classification",
                      metadata={
                          "help": "The downstream training task",
                          "choices": ["classification", "regression"]
                      })

    mlp_division: int = field(default=4,
                              metadata={
                                  'help': 'the ratio of the number of '
                                          'hidden dims in a current layer to the next MLP layer'
                              })
    combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat',
                                     metadata={
                                         'help': 'method to combine categorical and numerical features, '
                                                 'see README for all the method'
                                     })
    mlp_dropout: float = field(default=0.1,
                               metadata={
                                   'help': 'dropout ratio used for MLP layers'
                               })
    numerical_bn: bool = field(default=True,
                               metadata={
                                   'help': 'whether to use batchnorm on numerical features'
                               })
    use_simple_classifier: str = field(default=True,
                                       metadata={
                                           'help': 'whether to use single layer or MLP as final classifier'
                                       })
    mlp_act: str = field(default='relu',
                         metadata={
                             'help': 'the activation function to use for finetuning layers',
                             'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                         })
    gating_beta: float = field(default=0.2,
                               metadata={
                                   'help': "the beta hyperparameters used for gating tabular data "
                                           "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
                               })

    def __post_init__(self):
        assert self.column_info != self.column_info_path
        if self.column_info is None and self.column_info_path:
            with open(self.column_info_path, 'r') as f:
                self.column_info = json.load(f)


# ### Here are the data and training parameters we will use.
# For model we can specify any supported HuggingFace model classes (see README for more details) as well as any AutoModel that are from the supported model classes. For the data specifications, we need to specify a dictionary that specifies which columns are the `text` columns, `numerical feature` columns, `categorical feature` column, and the `label` column. If we are doing classification, we can also specify what each of the labels means in the label column through the `label list`. We can also specifiy these columns using a path to a json file with the argument `column_info_path` to `MultimodalDataTrainingArguments`.

# In[19]:

text_cols= ['Symbol']
# text_cols = ['Name','Symbol', 'Rating Agency Name', 'Sector']
# text_cols = ['secText']
# cat_cols = ['Symbol', 'Sector', 'CIK']
cat_cols = ['Sector', 'CIK'] + ['Name', 'Symbol', 'Rating Agency Name', 'Sector']
# cat_cols = ['Symbol', 'Sector', 'CIK'] + ['Name', 'Symbol', 'Rating Agency Name', 'Sector']
numerical_cols = ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding', 'netProfitMargin',
                  'pretaxProfitMargin', 'grossProfitMargin', 'operatingProfitMargin', 'returnOnAssets',
                  'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover', 'fixedAssetTurnover', 'debtEquityRatio',
                  'debtRatio', 'effectiveTaxRate', 'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare',
                  'cashPerShare', 'companyEquityMultiplier', 'ebitPerRevenue', 'enterpriseValueMultiple',
                  'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio', 'payablesTurnover']

column_info_dict = {
    'text_cols': text_cols,
    'num_cols': numerical_cols,
    'cat_cols': cat_cols,
    'label_col': 'Rating',
    'label_list': data_labels
}

model_args = ModelArguments(
    model_name_or_path='bert-base-uncased'
)

data_args = MultimodalDataTrainingArguments(
    data_path='..',
    combine_feat_method='gating_on_cat_and_num_feats_then_sum',
    column_info=column_info_dict,
    task='classification'
)

training_args = TrainingArguments(
    output_dir="../logs/model_name",
    logging_dir="../logs/runs",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=10,
    num_train_epochs=100,
    # evaluate_during_training=True,
    logging_steps=10,
    # eval_steps=50,
    dataloader_num_workers=16,
    fp16=True,
    evaluation_strategy='epoch',
    save_strategy='no'
)

set_seed(training_args.seed)

# ## Now we can load our model and data.
# ### We first instantiate our HuggingFace tokenizer
# This is needed to prepare our custom torch dataset. See `torch_dataset.py` for details.

# In[20]:


tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
print('Specified tokenizer: ', tokenizer_path_or_name)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path_or_name,
    cache_dir=model_args.cache_dir,
)

# ### Load dataset csvs to torch datasets
# The function `load_data_from_folder` expects a path to a folder that contains `train.csv`, `test.csv`, and/or `val.csv` containing the respective split datasets. 

# In[25]:


# Get Datasets
train_dataset, val_dataset, test_dataset = load_data_from_folder(
    data_args.data_path,
    data_args.column_info['text_cols'],
    tokenizer,
    label_col=data_args.column_info['label_col'],
    label_list=data_args.column_info['label_list'],
    categorical_cols=data_args.column_info['cat_cols'],
    numerical_cols=data_args.column_info['num_cols'],
    sep_text_token_str=tokenizer.sep_token,
)

# In[ ]:


num_labels = len(np.unique(train_dataset.labels))

# In[ ]:


config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)
tabular_config = TabularConfig(num_labels=num_labels,
                               cat_feat_dim=train_dataset.cat_feats.shape[1],
                               numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                               **vars(data_args))
config.tabular_config = tabular_config

# In[ ]:


model = AutoModelWithTabular.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    config=config,
    cache_dir=model_args.cache_dir
)

# ### We need to define a task-specific way of computing relevant metrics:

# In[ ]:


import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef, )


def calc_classification_metrics(p: EvalPrediction):
    pred_labels = np.argmax(p.predictions[0], axis=1)
    pred_scores = softmax(p.predictions[0], axis=1)[:, 1]
    labels = p.label_ids
    if len(np.unique(labels)) == 2:  # binary classification
        roc_auc_pred_score = roc_auc_score(labels, pred_scores)
        precisions, recalls, thresholds = precision_recall_curve(labels,
                                                                 pred_scores)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        fscore[np.isnan(fscore)] = 0
        ix = np.argmax(fscore)
        threshold = thresholds[ix].item()
        pr_auc = auc(recalls, precisions)
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        result = {'roc_auc': roc_auc_pred_score,
                  'threshold': threshold,
                  'pr_auc': pr_auc,
                  'recall': recalls[ix].item(),
                  'precision': precisions[ix].item(), 'f1': fscore[ix].item(),
                  'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item(), 'tp': tp.item()
                  }
    else:
        acc = (pred_labels == labels).mean()
        f1 = f1_score(y_true=labels, y_pred=pred_labels, average=None)
        result = {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
            "mcc": matthews_corrcoef(labels, pred_labels)
        }
        logger.info(result)
        cm = confusion_matrix(labels, pred_labels, )
        logger.info(f'\n{cm}')

        # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
        #                               display_labels=data_labels)
        # disp.plot()
        # plt.savefig(f'cm_{time.time()}.png')

    return result


# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=calc_classification_metrics,
)

# ## Launching the training is as simple is doing trainer.train() 🤗

# In[ ]:


# get_ipython().run_line_magic('', 'time')
trainer.train()

logger.info('################   test:')
trainer.evaluate(test_dataset)

#
# # ### Check that our training was successful using TensorBoard
#
# # In[ ]:
#
#
# # Load the TensorBoard notebook extension
# get_ipython().run_line_magic('load_ext', 'tensorboard')
#
#
# # In[ ]:
#
#
# get_ipython().run_line_magic('tensorboard', '--logdir./ logs / runs --port=6006')
