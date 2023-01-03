import argparse
from dataclasses import dataclass, field
import json
import os
from typing import Optional
import numpy as np
import pandas as pd
import torchinfo
from torch import optim
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed, TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback, AutoModel, HfArgumentParser
)
from transformers.training_args import TrainingArguments

import runner_setup
from arguments import BertModelArguments, MultimodalDataTrainingArguments, CrmmTrainingArguments, UNSUPERV_TASK
from metrics import calc_classification_metrics
from models.auto import AutoModelForCrmm
from models.category_autoencoder import CategoryAutoencoder, CategoryAutoencoderConfig
from models.multi_modal_dbn import MultiModalBertDBN, MultiModalConfig, CompatibleBert
from models.numerical_autoencoder import NumericalAutoencoder, NumericalAutoencoderConfig
from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig, AutoModelWithTabular

from transformers.utils import logging

logger = logging.get_logger('transformers')

os.environ['COMET_MODE'] = 'DISABLED'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# os.environ["WANDB_DISABLED"] = "true"


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


class CrmmTrainerCallback(TrainerCallback):
    crmm_clsif = None

    def __init__(self) -> None:
        super().__init__()

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        model = kwargs['model']
        # model.stop_dbn_training()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_begin(args, state, control, **kwargs)
        # print(f'\n****************start step: {state.global_step}***************')
        model = kwargs['model']
        # if model.training:
        #     for opt in dbn_optimizer:
        #         # Resets the optimizer
        #         opt.zero_grad()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        model = kwargs['model']
        a = 1
        # if state.global_step == 10:
        #     print()
        # print(f'***************end step: {state.global_step}***************')
        # print('\n\n')

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_epoch_begin(args, state, control, **kwargs)
        a = 1
        # print(f'\n=================start epoch: {state.epoch}=================')

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        a = 1
        # if state.epoch == 1:
        #     model = kwargs['model']
        #     model.stop_dbn_rbm_training()

        self.crmm_clsif.predict_on_test()
        # print(f'=================end epoch: {state.epoch}=================')


class CrmmClassification:

    def __init__(self, bert_model_args, data_args, training_args) -> None:
        self.bert_model_args = bert_model_args
        self.data_args = data_args
        self.training_args: CrmmTrainingArguments = training_args

        self.setup_tabular_cols()
        self.train_dataset, self.val_dataset, self.test_dataset, self.model = self.setup_model_with_dataset()
        self.trainer = self.create_trainer()

    def setup_tabular_cols(self):
        logger.info('setup_tabular_cols!')
        dataset_name = self.data_args.dataset_name
        if dataset_name in ['cr_sec', 'cr_sec_6']:
            text_cols = ['Name', 'Symbol', 'Rating Agency Name', 'Sector']
            # text_cols = ['secText']
            cat_cols = ['Symbol', 'Sector', 'CIK']
            # cat_cols = ['Symbol', 'Sector', 'CIK'] + ['Name', 'Symbol', 'Rating Agency Name', 'Sector']
            numerical_cols = ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding', 'netProfitMargin',
                              'pretaxProfitMargin', 'grossProfitMargin', 'operatingProfitMargin', 'returnOnAssets',
                              'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover', 'fixedAssetTurnover',
                              'debtEquityRatio',
                              'debtRatio', 'effectiveTaxRate', 'freeCashFlowOperatingCashFlowRatio',
                              'freeCashFlowPerShare',
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

        self.data_args.column_info = column_info_dict

    def setup_model_with_dataset(self):
        logger.info('setup_model_with_dataset!')
        # We first instantiate our HuggingFace tokenizer
        # This is needed to prepare our custom torch dataset. See `torch_dataset.py` for details.
        tokenizer_path_or_name = self.bert_model_args.tokenizer_name if self.bert_model_args.tokenizer_name \
            else self.bert_model_args.model_name_or_path
        # logger.info('Specified tokenizer: ', tokenizer_path_or_name)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path_or_name,
            cache_dir=self.bert_model_args.cache_dir,
            local_files_only=True if self.bert_model_args.cache_dir else False
        )

        # Get Datasets
        # !!! if the task is unsuperv pretrain or embedding prediction
        # , we merge train and val to `train` in load_data_from_folder
        train_dataset, val_dataset, test_dataset = load_data_from_folder(
            self.data_args.data_path,
            self.data_args.column_info['text_cols'],
            tokenizer,
            label_col=self.data_args.column_info['label_col'],
            label_list=self.data_args.column_info['label_list'],
            categorical_cols=self.data_args.column_info['cat_cols'],
            numerical_cols=self.data_args.column_info['num_cols'],
            sep_text_token_str=tokenizer.sep_token,
            # note: unsupervised training use all data
            merge_train_val_test=True if self.training_args.task in UNSUPERV_TASK else False,
        )

        num_labels = len(np.unique(train_dataset.labels))

        bert_config = AutoConfig.from_pretrained(
            self.bert_model_args.config_name if self.bert_model_args.config_name else self.bert_model_args.model_name_or_path,
            cache_dir=self.bert_model_args.cache_dir,
            local_files_only=True if self.bert_model_args.cache_dir else False
        )
        tabular_config = TabularConfig(num_labels=num_labels,
                                       cat_feat_dim=train_dataset.cat_feats.shape[1],
                                       numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                                       **vars(self.data_args))

        model = None
        if self.training_args.task == 'pretrain_num_ae_unsuperv':
            config = NumericalAutoencoderConfig(self.training_args.ae_embedding_dim, tabular_config)
            model = NumericalAutoencoder(config)
            # below not work
            # train_dataset = ConcatDataset([train_dataset, val_dataset])
            # val_dataset = None
        elif self.training_args.task == 'pretrain_cat_ae_unsuperv':
            config = CategoryAutoencoderConfig(self.training_args.ae_embedding_dim, tabular_config)
            model = CategoryAutoencoder(config)
        elif self.training_args.task == 'predict_num_ae_embedding':
            config = NumericalAutoencoderConfig.from_pretrained(self.training_args.num_ae_model_ckpt)
            model = AutoModelForCrmm.from_pretrained(self.training_args.num_ae_model_ckpt, config=config)
        elif self.training_args.task == 'predict_cat_ae_embedding':
            config = CategoryAutoencoderConfig.from_pretrained(self.training_args.cat_ae_model_ckpt)
            model = AutoModelForCrmm.from_pretrained(self.training_args.cat_ae_model_ckpt, config=config)
        # elif training_args.task == 'fit_multi_modal_dbn':
        #     config = MultiModalConfig(dbn_dataset_dir=training_args.ae_embedding_dataset_dir,
        #                                  # note!
        #                                  dbn_model_save_dir=training_args.output_dir,
        #                                  tabular_config=tabular_config,
        #                                  use_modality=training_args.modality,
        #                                  use_gpu=not training_args.no_cuda)
        #     model = MultiModalBertDBN(config)
        # elif training_args.task == 'fine_tune_only_dbns':
        #     config = MultiModalConfig(dbn_dataset_dir=training_args.ae_embedding_dataset_dir,
        #                                  # note!
        #                                  pretrained_dbn_model_dir=training_args.pretrained_dbn_model_dir,
        #                                  tabular_config=tabular_config,
        #                                  use_modality=training_args.modality,
        #                                  use_gpu=not training_args.no_cuda)
        #     model = MultiModalBertDBN(config)
        elif self.training_args.task == 'pretrain_multi_modal_dbn':
            config = MultiModalConfig(tabular_config=tabular_config,
                                      bert_config=bert_config,
                                      use_modality=self.training_args.modality,
                                      use_rbm_for_text=self.training_args.use_rbm_for_text,
                                      dbn_train_epoch=self.training_args.dbn_train_epoch,
                                      pretrain=True)
            model = MultiModalBertDBN.from_pretrained('bert-base-uncased', config=config)

        elif self.training_args.task == 'fine_tune_multi_modal_dbn_classification':
            bert_model = None
            config = MultiModalConfig.from_pretrained(self.training_args.pretrained_multi_modal_dbn_model_dir)
            if self.training_args.modality != config.use_modality:
                # if manually specific in training arguments command, override modality
                config.use_modality = self.training_args.modality
                bert_model = CompatibleBert.from_pretrained('bert-base-uncased', config.bert_config)
            config.pretrain = False  # note: remember to set to false, for fine tune
            model = MultiModalBertDBN.from_pretrained(self.training_args.pretrained_multi_modal_dbn_model_dir,
                                                      config=config)
            if bert_model is not None:
                model.bert = bert_model
            print()

        elif self.training_args.task == 'fine_tune_multi_modal_dbn_classification_scratch':
            config = MultiModalConfig(tabular_config=tabular_config,
                                      bert_config=bert_config,
                                      use_modality=self.training_args.modality,
                                      dbn_train_epoch=self.training_args.dbn_train_epoch,
                                      pretrain=False)
            model = MultiModalBertDBN.from_pretrained('bert-base-uncased', config=config)

        # logger.info("Model:\n{}".format(model))
        # logger.info("Model config:\n{}".format(config))
        torchinfo.summary(model)

        return train_dataset, val_dataset, test_dataset, model

    def create_trainer(self):
        def get_metric_fn(task):
            if 'classification' in task:
                return calc_classification_metrics
            else:
                # None will use loss as default metric
                return None

        trainer_callbacks = [
            EarlyStoppingCallback(early_stopping_patience=self.training_args.patience),
            TrainerLoggerCallback(),
            CrmmTrainerCallback()
        ]

        if self.training_args.task == 'pretrain_multi_modal_dbn':
            """
            to pretrain (fit) rbm, dbm,  only need to train 1 epoch of (MultiModalBertDBN),
            in forward, it will fit rbm or for epoch of dbn self.config.rbm_train_epoch
            """
            self.training_args.num_train_epochs = 1

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            # train autoencoder will eval on test set since it is merge_train_val
            eval_dataset=self.test_dataset if self.val_dataset is None else self.val_dataset,
            compute_metrics=get_metric_fn(self.training_args.task),
            callbacks=trainer_callbacks,
            # optimizers=[NormalOptmi]
        )

        trainer_callbacks[2].crmm_clsif = self

        return trainer

    def train(self):
        # if self.training_args.task == 'fit_multi_modal_dbn':
        #     self.model.fit_dbns()
        # elif self.training_args.task == 'fine_tune_only_dbns':
        #     self.model.manually_fine_tune_only_dbns()
        # else:
        #     self.trainer.train()
        if self.training_args.task == 'pretrain_multi_modal_dbn':
            self.manually_pretrain_dbn()
        else:
            self.trainer.train()

    def manually_pretrain_dbn(self):
        train_dataloader = self.trainer.get_train_dataloader()
        self.model.train()
        steps_trained_progress_bar = tqdm(total=len(train_dataloader))
        for step, inputs in enumerate(train_dataloader):
            inputs = self.trainer._prepare_inputs(inputs)
            outputs = self.model(**inputs)
            steps_trained_progress_bar.update()
        self.trainer.save_model()

    def predict_embedding(self):
        if self.training_args.do_predict and self.training_args.task in ['predict_num_ae_embedding',
                                                                         'predict_cat_ae_embedding']:
            # !!! note predict embedding on entire dataset and save, the embedding dataset is used to train dbn
            predictions = self.trainer.predict(test_dataset=self.train_dataset).predictions
            np.save(
                os.path.join(self.training_args.ae_embedding_dataset_dir, f'{self.training_args.task}_results.npy'),
                predictions)
            np.save(os.path.join(self.training_args.ae_embedding_dataset_dir, f'labels.npy'),
                    self.train_dataset.labels)
            # emb_train, emb_val, emb_test = np.split(predictions,
            #                                         [int(.8 * len(predictions)), int(.9 * len(predictions))])
            # np.save(f'{training_args.task}_results_train.npy', emb_train)
            # np.save(f'{training_args.task}_results_val.npy', emb_val)
            # np.save(f'{training_args.task}_results_test.npy', emb_test)

    def predict_on_test(self):
        logger.info('################   test:')
        self.trainer.predict(self.test_dataset)


if __name__ == '__main__':
    parser = HfArgumentParser([BertModelArguments, MultimodalDataTrainingArguments, CrmmTrainingArguments])
    _bert_model_args, _data_args, _training_args = parser.parse_args_into_dataclasses()
    _training_args = runner_setup.setup(_training_args)

    cc = CrmmClassification(_bert_model_args, _data_args, _training_args)
    cc.train()
