import os

import numpy as np
import torchinfo
from torch.utils.data import ConcatDataset
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


def main(bert_model_args: BertModelArguments,
         data_args: MultimodalDataArguments,
         training_args: CrmmTrainingArguments):
    task = training_args.task
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
            # 'text_cols': ['secText', 'secKeywords'],
            'text_cols': ['secKeywords'],
            # 'text_cols': ['secText'],
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
    train_dataset, test_dataset, val_dataset = mm_data.get_datasets()
    n_labels = len(np.unique(train_dataset.labels))
    if task == 'pretrain':
        # if pretrain task, concat train and val to unsupervised train!
        train_dataset = ConcatDataset([train_dataset, val_dataset])
        val_dataset = None
    nunique_cat_nums, cat_emb_dims = mm_data.get_nunique_cat_nums_and_emb_dim(equal_dim=None)

    # @@@@ 3. MODEL
    bert_params = {'pretrained_model_name_or_path': bert_model_args.bert_model_name,
                   'cache_dir': bert_model_args.cache_dir,
                   'local_files_only': bert_model_args.load_hf_model_from_cache}
    tokenizer = AutoTokenizer.from_pretrained(**bert_params)
    default_model_config_params = dict(n_labels=n_labels,
                                       num_feat_dim=test_dataset.numerical_feats.shape[1],
                                       nunique_cat_nums=nunique_cat_nums,
                                       cat_emb_dims=cat_emb_dims,
                                       use_modality=training_args.use_modality,
                                       bert_params=bert_params,
                                       bert_model_name=bert_model_args.bert_model_name)
    if task == 'pretrain':
        model_config = MultiModalModelConfig(**default_model_config_params,
                                             use_hf_pretrained_bert=False,
                                             pretrained=False)
        model = MultiModalDBN(model_config)
    elif task == 'fine_tune':
        # finetune, model load from saved dir
        model_config = MultiModalModelConfig.from_pretrained(training_args.pretrained_model_dir)
        model_config.use_hf_pretrained_bert = False
        model = MultiModalForClassification.from_pretrained(training_args.pretrained_model_dir, config=model_config)
    elif task == 'fine_tune_from_scratch':
        model_config = MultiModalModelConfig(**default_model_config_params,
                                             use_hf_pretrained_bert=False,
                                             pretrained=True)  # manually set pretrained to True!
        # create model, where the bert model is loaded from hf model in models.rbm_factory.RBMFactory._get_bert
        model = MultiModalForClassification(model_config)
    else:
        raise ValueError(f'Unknown task: {task}')
    logger.info(f'\n{model}')
    logger.info(f'\n{torchinfo.summary(model, verbose=0)}')

    # @@@@ 4. TRAINING
    get_trainer = lambda model: Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # can be None in pretrain
        compute_metrics=calc_classification_metrics if 'fine_tune' in task else None,
        callbacks=[EarlyStoppingCallback(training_args.patience)] if 'fine_tune' in task else None,
        data_collator=MultimodalDatasetCollator(tokenizer, bert_model_args.max_seq_length)
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
        # should set pretrained to True. Hence, we can later continue to finetune the model
        model.config.pretrained = True
        trainer.save_model()
    else:
        trainer = get_trainer(model)
        trainer.train()
        trainer.save_model()
    logger.info(f"Best model path: {trainer.state.best_model_checkpoint}")
    logger.info(f"Best metric value: {trainer.state.best_metric}")

    # @@@@ 5. EVALUATION
    trainer.predict(test_dataset)


if __name__ == '__main__':
    parser = HfArgumentParser([BertModelArguments, MultimodalDataArguments, CrmmTrainingArguments])
    _bert_model_args, _data_args, _training_args = parser.parse_args_into_dataclasses()
    _training_args = runner_setup.setup(_training_args)
    logger.info(f'training_args: {_training_args}')

    main(_bert_model_args, _data_args, _training_args)
