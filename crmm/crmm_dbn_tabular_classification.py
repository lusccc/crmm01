import os

import numpy as np
import torchinfo
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback, HfArgumentParser
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging

import runner_setup
from arguments import BertModelArguments, MultimodalDataArguments, CrmmTrainingArguments
from dataset.multimodal_data import MultimodalData
from metrics import calc_classification_metrics
from models.multi_modal_dbn import MultiModalDBN, MultiModalModelConfig, CompatibleBert
from dataset.multimodal_dataset import MultimodalDatasetCollator
from models.rbm_factory import RBMFactory
from models.tabular_config import TabularConfig

logger = logging.get_logger('transformers')

os.environ['COMET_MODE'] = 'DISABLED'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["WANDB_DISABLED"] = "true"


class TrainerLoggerCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        # A bare [`TrainerCallback`] that just prints the logs.
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)


class CrmmTrainerCallback(TrainerCallback):

    def __init__(self, runner) -> None:
        self.runner = runner

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        self.runner.predict_on_test()


class CrmmClassification:

    def __init__(self, bert_model_args, data_args, training_args) -> None:
        self.trainer = None
        self.bert_config = None
        self.tabular_config = None
        self.tokenizer = None
        self.model = None
        self.test_dataset = None
        self.val_dataset = None
        self.crmm_data = None
        self.train_dataset = None

        self.bert_model_args = bert_model_args
        self.data_args = data_args
        self.training_args = training_args

        self.prepare()

    def prepare(self):
        # @@@@ 1. TABULAR COLUMNS
        dataset_name = self.data_args.dataset_name
        if dataset_name in ['cr_sec', 'cr_sec_6']:
            if dataset_name == 'cr_sec':
                label_list = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC']
            elif dataset_name == 'cr_sec_6':
                label_list = ['AA+', 'A', 'BBB', 'BB', 'B', 'CCC-']
            else:
                label_list = None
            # note: the num and cat cols will be automatically inferred in `data.crmm_data.MultimodalData`
            column_info_dict = {
                'text_cols': ['secText', 'secKeywords'],
                'label_col': 'Rating',
                'label_list': label_list
            }
        else:
            column_info_dict = None
        self.data_args.column_info = column_info_dict

        # @@@@ 2. TOKENIZER
        tokenizer_path_or_name = self.bert_model_args.tokenizer_name if self.bert_model_args.tokenizer_name \
            else self.bert_model_args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path_or_name,
            cache_dir=self.bert_model_args.cache_dir,
            local_files_only=self.bert_model_args.load_hf_model_from_cache
        )
        self.tokenizer = tokenizer

        # @@@@ 3. DATASET
        self.mm_data = MultimodalData(self.data_args,
                                        self.data_args.column_info['label_col'],
                                        self.data_args.column_info['label_list'],
                                        self.data_args.column_info['text_cols'],
                                        num_transform_method=self.data_args.numerical_transformer_method)
        self.train_dataset, self.val_dataset, self.test_dataset = self.mm_data.get_datasets()

        # @@@@ 4. CONFIGS
        num_labels = len(np.unique(self.train_dataset.labels))
        cat_nums, cat_emb_dims = self.mm_data.get_cat_num_and_emb_dim()
        self.tabular_config = TabularConfig(n_labels=num_labels,
                                            cat_feat_dim=self.train_dataset.cat_feats.shape[1],
                                            num_feat_dim=self.train_dataset.numerical_feats.shape[1],
                                            n_cat=cat_nums,
                                            cat_emb_dims=cat_emb_dims)
        self.bert_config = AutoConfig.from_pretrained(
            self.bert_model_args.config_name if self.bert_model_args.config_name else self.bert_model_args.model_name_or_path,
            cache_dir=self.bert_model_args.cache_dir,
            local_files_only=self.bert_model_args.load_hf_model_from_cache
        )

        # @@@@ 5. MODEL
        task = self.training_args.task
        model = None
        if task in 'ml_comparison':
            model
        else:
            if task == 'pretrain_multi_modal_dbn':
                config = MultiModalModelConfig(tabular_config=self.tabular_config,
                                               bert_config=self.bert_config,
                                               pretrain=True)
                model = MultiModalDBN(config)
                model = MultiModalDBN.from_pretrained('bert-base-uncased', config=config,
                                                      local_files_only=self.bert_model_args.load_hf_model_from_cache)

                rbm_factory = RBMFactory()

            elif task == 'fine_tune_multi_modal_dbn_classification':
                bert_model = None
                config = MultiModalModelConfig.from_pretrained(self.training_args.pretrained_multi_modal_dbn_model_dir)
                # if manually specific in training arguments command, override modalities
                if self.training_args.modalities != config.use_modality:
                    config.use_modality = self.training_args.modalities
                    bert_model = CompatibleBert.from_pretrained('bert-base-uncased', config.bert_config,
                                                                local_files_only=self.bert_model_args.load_hf_model_from_cache)
                config.pretrain = False  # note: remember to set to false, for fine tune
                model = MultiModalDBN.from_pretrained(self.training_args.pretrained_multi_modal_dbn_model_dir,
                                                      config=config)
                if bert_model is not None:
                    model.bert = bert_model
            elif task == 'fine_tune_multi_modal_dbn_classification_scratch':
                config = MultiModalModelConfig(tabular_config=self.tabular_config,
                                               bert_config=self.bert_config,
                                               use_modality=self.training_args.modalities,
                                               use_rbm_for_text=self.training_args.use_rbm_for_text,
                                               dbn_train_epoch=self.training_args.dbn_train_epoch,
                                               pretrain=False)
                model = MultiModalDBN.from_pretrained('bert-base-uncased', config=config,
                                                      local_files_only=self.bert_model_args.load_hf_model_from_cache)

            model.set_tokenizer(self.tokenizer)
            self.model = model
            torchinfo.summary(model)

        # @@@@ 6. TRAINER
        get_trainer = lambda model: Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset if self.val_dataset is None else self.val_dataset,
            compute_metrics=calc_classification_metrics if 'classification' in task else None,
            callbacks=[
                EarlyStoppingCallback(self.training_args.patience),
                TrainerLoggerCallback(),
                CrmmTrainerCallback(self)
            ],
            data_collator=MultimodalDatasetCollator(self.tokenizer)
        )

        if task == 'pretrain_multi_modal_dbn':
            for m, rbm in model.rbms.items():
                if rbm is not None:
                    trainer = get_trainer(rbm)
                    trainer.train()
                    model.rbms[m] = rbm

        if self.training_args.task == 'pretrain_multi_modal_dbn':
            """
            to pretrain (fit) rbm, dbm,  only need to train 1 epoch of (MultiModalDBN),
            in forward, it will fit rbm or for epoch of dbn `self.config.rbm_train_epoch`
            """
            self.training_args.num_train_epochs = 1

    def train(self):
        if self.training_args.task == 'pretrain_multi_modal_dbn':
            self.manually_pretrain_dbn()
        else:
            self.trainer.train()

    def manually_pretrain_dbn(self):
        train_dataloader = self.trainer.get_train_dataloader()
        self.model.train()
        steps_trained_progress_bar = tqdm(total=len(train_dataloader))
        for step, inputs in enumerate(train_dataloader):
            if step == 100:
                print()
            inputs = self.trainer._prepare_inputs(inputs)
            outputs = self.model(**inputs)
            steps_trained_progress_bar.update()
        self.trainer.save_model()

    def predict_on_test(self):
        logger.info('@@@@@@@@@@ predict_on_test: @@@@@@@@@')
        self.trainer.predict(self.test_dataset)


if __name__ == '__main__':
    parser = HfArgumentParser([BertModelArguments, MultimodalDataArguments, CrmmTrainingArguments])
    _bert_model_args, _data_args, _training_args = parser.parse_args_into_dataclasses()
    _training_args = runner_setup.setup(_training_args)
    logger.info(f'training_args: {_training_args}')

    cc = CrmmClassification(_bert_model_args, _data_args, _training_args)
    cc.train()
