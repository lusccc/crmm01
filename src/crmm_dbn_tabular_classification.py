import argparse
from dataclasses import dataclass, field
import json
import os
from typing import Optional
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import ConcatDataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed, TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback, AutoModel, HfArgumentParser
)
from transformers.training_args import TrainingArguments

import runner_setup
from arguments import BertModelArguments, MultimodalDataTrainingArguments, CrmmTrainingArguments, UNSUPERV_TASK, \
    PREDICTION_TASK, DBN_TASK
from metrics import calc_classification_metrics
from models.auto import AutoModelForCrmm
from models.category_autoencoder import CategoryAutoencoder, CategoryAutoencoderConfig
from models.multi_modal_dbn import MultiModalDBN, MultiModalDBNConfig
from models.numerical_autoencoder import NumericalAutoencoder, NumericalAutoencoderConfig
from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig, AutoModelWithTabular

from transformers.utils import logging

logger = logging.get_logger('transformers')

os.environ['COMET_MODE'] = 'DISABLED'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


#  used in forward pass
# dbn_optimizer = [optim.Adam(m.parameters(), lr=0.001) for m in model.cat_feat_dbn.models] + \
#                 [optim.Adam(m.parameters(), lr=0.001) for m in model.numerical_feat_dbn.models] + \
#                 [optim.Adam(m.parameters(), lr=0.001) for m in model.joint_dbn.models]
class TrainerLoggerCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)


class DBNTrainCallback(TrainerCallback):



    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        model = kwargs['model']
        model.stop_dbn_training()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_begin(args, state, control, **kwargs)
        model = kwargs['model']
        # if model.training:
        #     for opt in dbn_optimizer:
        #         # Resets the optimizer
        #         opt.zero_grad()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        model = kwargs['model']
        a = 1
        # if model.training:
        #     for opt in dbn_optimizer:
        #         # Performs the gradient update
        #         opt.step()
        # print()

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_epoch_begin(args, state, control, **kwargs)
        a = 1

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        a = 1
        # if state.epoch == 5:
        #     model = kwargs['model']
        #     model.stop_dbn_training()


def eval_model(trainer, eval_dataset):
    logger.info('################   test:')
    trainer.evaluate(eval_dataset)


def train_model(model, train_dataset, val_dataset, test_dataset):
    def get_metric_fn(task):
        if 'classification' in task:
            return calc_classification_metrics
        else:
            # None will use loss as default metric
            return None

    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        # train autoencoder will eval on test set since it is merge_train_val
        eval_dataset=test_dataset if val_dataset is None else val_dataset,
        compute_metrics=get_metric_fn(TRAINING_ARGS.task),
        callbacks=[
            # EarlyStoppingCallback(early_stopping_patience=TRAINING_ARGS.patience),
            TrainerLoggerCallback(),
            DBNTrainCallback()
        ],
        # optimizers=[NormalOptmi]
    )
    if TRAINING_ARGS.do_predict:
        if TRAINING_ARGS.task in ['predict_num_ae_embedding', 'predict_cat_ae_embedding']:
            # !!! note predict embedding on entire dataset and save, the embedding dataset is used to train dbn
            predictions = trainer.predict(test_dataset=train_dataset).predictions
            np.save(os.path.join(TRAINING_ARGS.ae_embedding_dataset_dir, f'{TRAINING_ARGS.task}_results.npy'),
                    predictions)
            np.save(os.path.join(TRAINING_ARGS.ae_embedding_dataset_dir, f'labels.npy'), train_dataset.labels)
            # emb_train, emb_val, emb_test = np.split(predictions,
            #                                         [int(.8 * len(predictions)), int(.9 * len(predictions))])
            # np.save(f'{TRAINING_ARGS.task}_results_train.npy', emb_train)
            # np.save(f'{TRAINING_ARGS.task}_results_val.npy', emb_val)
            # np.save(f'{TRAINING_ARGS.task}_results_test.npy', emb_test)

    elif TRAINING_ARGS.task == 'fit_multi_modal_dbn':
        model.fit_dbns()
    elif TRAINING_ARGS.task == 'fine_tune_only_dbns':
        model.manually_fine_tune_only_dbns()
    else:
        trainer.train()
    return trainer


def setup_model_with_dataset():
    logger.info('setup_model_with_dataset!')
    # We first instantiate our HuggingFace tokenizer
    # This is needed to prepare our custom torch dataset. See `torch_dataset.py` for details.
    tokenizer_path_or_name = BERT_MODEL_ARGS.tokenizer_name if BERT_MODEL_ARGS.tokenizer_name \
        else BERT_MODEL_ARGS.model_name_or_path
    # logger.info('Specified tokenizer: ', tokenizer_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path_or_name,
        cache_dir=BERT_MODEL_ARGS.cache_dir,
        local_files_only=True if BERT_MODEL_ARGS.cache_dir else False
    )

    # Get Datasets
    # !!! if the task is unsuperv pretrain or embedding prediction
    # , we merge train and val to `train` in load_data_from_folder
    train_dataset, val_dataset, test_dataset = load_data_from_folder(
        DATA_ARGS.data_path,
        DATA_ARGS.column_info['text_cols'],
        tokenizer,
        label_col=DATA_ARGS.column_info['label_col'],
        label_list=DATA_ARGS.column_info['label_list'],
        categorical_cols=DATA_ARGS.column_info['cat_cols'],
        numerical_cols=DATA_ARGS.column_info['num_cols'],
        sep_text_token_str=tokenizer.sep_token,
        ###  note: unsupervised training using train+val as train set, monitor on test
        merge_train_val=True if TRAINING_ARGS.task in UNSUPERV_TASK else False,
        ###  note: ae predict using entire dataset
        merge_train_val_test=True if TRAINING_ARGS.task in PREDICTION_TASK else False,
    )

    num_labels = len(np.unique(train_dataset.labels))

    bert_config = AutoConfig.from_pretrained(
        BERT_MODEL_ARGS.config_name if BERT_MODEL_ARGS.config_name else BERT_MODEL_ARGS.model_name_or_path,
        cache_dir=BERT_MODEL_ARGS.cache_dir,
        local_files_only=True if BERT_MODEL_ARGS.cache_dir else False
    )
    tabular_config = TabularConfig(num_labels=num_labels,
                                   cat_feat_dim=train_dataset.cat_feats.shape[1],
                                   numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                                   **vars(DATA_ARGS))
    bert_config.tabular_config = tabular_config

    # model = AutoModelWithTabular.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     config=bert_config,
    #     cache_dir=model_args.cache_dir
    # )
    # model = MultiModalDBN(tabular_config, 128, 128, dbn_train_epoch=3)
    model = None
    if TRAINING_ARGS.task == 'pretrain_num_ae_unsuperv':
        config = NumericalAutoencoderConfig(TRAINING_ARGS.ae_embedding_dim, tabular_config)
        model = NumericalAutoencoder(config)
        # below not work
        # train_dataset = ConcatDataset([train_dataset, val_dataset])
        # val_dataset = None
    elif TRAINING_ARGS.task == 'pretrain_cat_ae_unsuperv':
        config = CategoryAutoencoderConfig(TRAINING_ARGS.ae_embedding_dim, tabular_config)
        model = CategoryAutoencoder(config)
    elif TRAINING_ARGS.task == 'predict_num_ae_embedding':
        config = NumericalAutoencoderConfig.from_pretrained(TRAINING_ARGS.num_ae_model_ckpt)
        model = AutoModelForCrmm.from_pretrained(TRAINING_ARGS.num_ae_model_ckpt, config=config)
    elif TRAINING_ARGS.task == 'predict_cat_ae_embedding':
        config = CategoryAutoencoderConfig.from_pretrained(TRAINING_ARGS.cat_ae_model_ckpt)
        model = AutoModelForCrmm.from_pretrained(TRAINING_ARGS.cat_ae_model_ckpt, config=config)
    # elif TRAINING_ARGS.task == 'fit_multi_modal_dbn':
    #     config = MultiModalDBNConfig(dbn_dataset_dir=TRAINING_ARGS.ae_embedding_dataset_dir,
    #                                  # note!
    #                                  dbn_model_save_dir=TRAINING_ARGS.output_dir,
    #                                  tabular_config=tabular_config,
    #                                  use_modality=TRAINING_ARGS.modality,
    #                                  use_gpu=not TRAINING_ARGS.no_cuda)
    #     model = MultiModalDBN(config)
    # elif TRAINING_ARGS.task == 'fine_tune_only_dbns':
    #     config = MultiModalDBNConfig(dbn_dataset_dir=TRAINING_ARGS.ae_embedding_dataset_dir,
    #                                  # note!
    #                                  pretrained_dbn_model_dir=TRAINING_ARGS.pretrained_dbn_model_dir,
    #                                  tabular_config=tabular_config,
    #                                  use_modality=TRAINING_ARGS.modality,
    #                                  use_gpu=not TRAINING_ARGS.no_cuda)
    #     model = MultiModalDBN(config)
    elif TRAINING_ARGS.task == 'multi_modal_dbn_ensemble_classification':
        config = MultiModalDBNConfig(tabular_config=tabular_config,
                                     use_modality=TRAINING_ARGS.modality,
                                     dbn_train_epoch=TRAINING_ARGS.dbn_train_epoch,
                                     use_gpu=not TRAINING_ARGS.no_cuda)
        model = MultiModalDBN(config)

    logger.info("Model:\n{}".format(model))
    # logger.info("Model config:\n{}".format(config))

    return train_dataset, val_dataset, test_dataset, model


def setup_tabular_cols(dataset_name):
    logger.info('setup_tabular_cols!')
    if dataset_name in ['cr_sec', 'cr_sec_6']:
        text_cols = ['Name', 'Symbol', 'Rating Agency Name', 'Sector']
        # text_cols = ['secText']
        cat_cols = ['Symbol', 'Sector', 'CIK']
        # cat_cols = ['Symbol', 'Sector', 'CIK'] + ['Name', 'Symbol', 'Rating Agency Name', 'Sector']
        numerical_cols = ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding', 'netProfitMargin',
                          'pretaxProfitMargin', 'grossProfitMargin', 'operatingProfitMargin', 'returnOnAssets',
                          'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover', 'fixedAssetTurnover',
                          'debtEquityRatio',
                          'debtRatio', 'effectiveTaxRate', 'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare',
                          'cashPerShare', 'companyEquityMultiplier', 'ebitPerRevenue', 'enterpriseValueMultiple',
                          'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio', 'payablesTurnover']
        if dataset_name == 'cr_sec':
            label_list = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC']
        elif dataset_name == 'cr_sec_6':
            label_list = ['AA+', 'A', 'BBB', 'BB', 'B', 'CCC-']
        column_info_dict = {
            'text_cols': text_cols,
            'num_cols': numerical_cols,
            'cat_cols': cat_cols,
            'label_col': 'Rating',
            'label_list': label_list
        }

    DATA_ARGS.column_info = column_info_dict


def main():
    setup_tabular_cols(DATA_ARGS.dataset_name)
    train_dataset, val_dataset, test_dataset, model = setup_model_with_dataset()
    trainer = train_model(model, train_dataset, val_dataset, test_dataset)
    # eval_model(trainer, test_dataset)


if __name__ == '__main__':
    parser = HfArgumentParser([BertModelArguments, MultimodalDataTrainingArguments, CrmmTrainingArguments])
    BERT_MODEL_ARGS, DATA_ARGS, TRAINING_ARGS = parser.parse_args_into_dataclasses()
    TRAINING_ARGS = runner_setup.setup(TRAINING_ARGS)

    main()
