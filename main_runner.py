import os

import numpy as np
import torchinfo
from transformers import AutoTokenizer, Trainer, EarlyStoppingCallback, HfArgumentParser

from crmm import runner_setup
from crmm.arguments import BertModelArguments, MultimodalDataArguments, CrmmTrainingArguments
from transformers.utils import logging

from crmm.dataset.multimodal_data import MultimodalData
from crmm.dataset.multimodal_dataset import MultimodalDatasetCollator
from crmm.metrics import calc_classification_metrics
from crmm.models.multi_modal_dbn import MultiModalModelConfig, MultiModalDBN, MultiModalForClassification
from crmm.runner_callback import TrainerLoggerCallback, CrmmTrainerCallback

logger = logging.get_logger('transformers')

os.environ['COMET_MODE'] = 'DISABLED'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["WANDB_DISABLED"] = "true"


def main(bert_model_args: BertModelArguments, data_args: MultimodalDataArguments, training_args: CrmmTrainingArguments):
    # @@@@ 1. TABULAR COLUMNS
    dataset_name = data_args.dataset_name
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
    data_args.column_info = column_info_dict

    # @@@@ 2. DATASET
    mm_data = MultimodalData(data_args,
                             data_args.column_info['label_col'],
                             data_args.column_info['label_list'],
                             data_args.column_info['text_cols'],
                             num_transform_method=data_args.numerical_transformer_method)
    train_dataset, val_dataset, test_dataset = mm_data.get_datasets()
    num_labels = len(np.unique(train_dataset.labels))
    cat_nums, cat_emb_dims = mm_data.get_cat_num_and_emb_dim()

    # @@@@ 3. MODEL
    task = training_args.task
    bert_params = {'pretrained_model_name_or_path': bert_model_args.bert_model_name,
                   'cache_dir': bert_model_args.cache_dir,
                   'local_files_only': bert_model_args.load_hf_model_from_cache}
    tokenizer = AutoTokenizer.from_pretrained(**bert_params)
    if 'pretrain' in task or 'scratch' in task:
        model_config = MultiModalModelConfig(n_labels=num_labels,
                                             cat_feat_dim=train_dataset.cat_feats.shape[1],
                                             num_feat_dim=train_dataset.numerical_feats.shape[1],
                                             n_cat=cat_nums,
                                             cat_emb_dims=cat_emb_dims,
                                             use_modality=training_args.use_modality,
                                             bert_params=bert_params,
                                             pretrained=False)
        # create model, where the bert model is loaded from hf model in models.rbm_factory.RBMFactory._get_bert
        model = MultiModalDBN(model_config)
    else:
        # finetune, model load from saved dir
        model_config = MultiModalModelConfig.from_pretrained(training_args.pretrained_model_dir)
        model = MultiModalForClassification.from_pretrained(training_args.pretrained_model_dir, config=model_config)
    torchinfo.summary(model)

    # @@@@ 4. TRAINING
    get_trainer = lambda model: Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if val_dataset is None else val_dataset,
        compute_metrics=calc_classification_metrics if 'fine_tune' in task else None,
        callbacks=[
            EarlyStoppingCallback(training_args.patience),
            TrainerLoggerCallback(),
        ],
        data_collator=MultimodalDatasetCollator(tokenizer)
    )
    if task == 'pretrain':
        # 1. pretrain each rbm of each modality
        model.set_pretrain_target('modality_rbm')
        trainer = get_trainer(model)
        trainer.train()
        # 2. if more than 1 modality, continue to pretrain joint rbm
        if len(model_config.use_modality) > 1:
            model.set_pretrain_target('joint_rbm')
            trainer.train()
        # should set pretrained to True. Hence, we can continue to finetune the model
        model.config.pretrained = True
        trainer.save_model()
    else:
        trainer = get_trainer(model)
        trainer.train()
        trainer.save_model()


if __name__ == '__main__':
    parser = HfArgumentParser([BertModelArguments, MultimodalDataArguments, CrmmTrainingArguments])
    _bert_model_args, _data_args, _training_args = parser.parse_args_into_dataclasses()
    _training_args = runner_setup.setup(_training_args)
    logger.info(f'training_args: {_training_args}')

    main(_bert_model_args, _data_args, _training_args)
