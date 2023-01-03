import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments, IntervalStrategy

UNSUPERV_TASK = ['pretrain_num_ae_unsuperv', 'pretrain_cat_ae_unsuperv', 'pretrain_multi_modal_dbn']
SUPERV_TASK = ['fine_tune_multi_modal_dbn_classification', 'fine_tune_multi_modal_dbn_classification_scratch']
PREDICTION_TASK = ['predict_num_ae_embedding', 'predict_cat_ae_embedding']
TASK = UNSUPERV_TASK + PREDICTION_TASK + SUPERV_TASK


@dataclass
class CrmmTrainingArguments(TrainingArguments):
    # !!!!OURS!!!!
    report_to: str = field(default='wandb')
    root_dir: str = field(default='./exps', metadata={"help": "parent dir of output_dir"})
    task: str = field(default="fine_tune_multi_modal_dbn_classification", metadata={
        "help": "", 'choices': TASK})
    modality: str = field(default="num,cat,text", metadata={
        "help": "used in multi dbn"})
    use_rbm_for_text: bool = field(default=True, metadata={
        "help": ""})
    # num_ae_model_ckpt: str = field(
    #     default='exps/pretrain_num_ae_unsuperv_2022-11-01_14-31-39_w9p/output/checkpoint-1488',
    #     metadata={"help": "num dbn ae model path, used for ae embedding prediction"})
    # cat_ae_model_ckpt: str = field(
    #     default='exps/pretrain_cat_ae_unsuperv_2022-11-02_14-34-14_JxN/output/checkpoint-6',
    #     metadata={"help": "cat dbn ae model path, used for ae embedding prediction"})
    # ae_embedding_dataset_dir: str = field(default='data/cr_sec_ae_embedding',
    #                                       metadata={"help": "dir to save embedding prediction results"})
    # ae_embedding_dim: str = field(default=768, metadata={"help": ""})
    pretrained_multi_modal_dbn_model_dir: str = field(
        default='exps/pretrain_multi_modal_dbn_2023-01-02_20-10-18_HKc/output',
        metadata={"help": ""})
    dbn_train_epoch: int = field(default=5, metadata={"help": ""})
    patience: int = field(default=1000, metadata={"help": ""})


    # !!!!BELOW ARE HUGGINGFACE ARGS!!!!
    # output_dir and logging_dir will be auto set in runner_setup.setup
    output_dir: str = field(default=None, metadata={
        "help": "The output directory where the model predictions and checkpoints will be written."}, )
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    overwrite_output_dir: bool = field(default=True, metadata={
        "help": ("Overwrite the content of the output directory. "
                 "Use this to continue training if output_dir points to a checkpoint directory.")})
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    # ---
    num_train_epochs: float = field(default=250, metadata={"help": "Total number of training epochs to perform."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    # ---
    per_device_train_batch_size: int = field(default=7, metadata={
        "help": "Batch size per GPU/TPU core/CPU for training."})
    evaluation_strategy: Union[IntervalStrategy, str] = field(default="epoch", metadata={
        "help": "The evaluation strategy to use."}, )
    dataloader_num_workers: int = field(default=12, metadata={
        "help": ("Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                 " in the main process.")})
    save_strategy: Union[IntervalStrategy, str] = field(default="epoch", metadata={
        "help": "The checkpoint save strategy to use."})
    load_best_model_at_end: Optional[bool] = field(default=True, metadata={
        "help": "Whether or not to load the best model found during training at the end of training."})
    save_total_limit: Optional[int] = field(default=2, metadata={
        "help": ("Limit the total amount of checkpoints. "
                 "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints")})
    auto_find_batch_size: bool = field(default=True, metadata={
        "help": ("Whether to automatically decrease the batch size in half and rerun the training loop again each time"
                 " a CUDA Out-of-Memory was reached")})
    metric_for_best_model: str = field(default='eval_acc') # used for early stopping
    greater_is_better: bool = field(default=True) # used for early stopping

    def __post_init__(self):
        super().__post_init__()
        if self.task in PREDICTION_TASK:
            self.do_predict = True
        self.modality = self.modality.split(',')
        self.modality = {
            'num': True if 'num' in self.modality else False,
            'cat': True if 'cat' in self.modality else False,
            'text': True if 'text' in self.modality else False,
        }


@dataclass
class BertModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(default='bert-base-uncased', metadata={
        "help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(default=None, metadata={
        "help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={
        "help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default='./exps/model_config_cache', metadata={
        "help": "Where do you want to store the pretrained models downloaded from s3"})


@dataclass
class MultimodalDataTrainingArguments:
    """
    Arguments pertaining to how we combine tabular features
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: str = field(default='cr_sec_6', metadata={"help": ""})
    data_path: str = field(default=f'./data/cr_sec_6', metadata={
        'help': 'the path to the csv file containing the dataset'})
    column_info_path: str = field(default=None, metadata={
        'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'})
    column_info: dict = field(default=None, metadata={
        'help': 'a dict referencing the text, categorical, numerical, and label columns'
                'its keys are text_cols, num_cols, cat_cols, and label_col'})
    categorical_encode_type: str = field(default='ohe', metadata={
        'help': 'sklearn encoder to use for categorical data',
        'choices': ['ohe', 'binary', 'label', 'none']})
    numerical_transformer_method: str = field(default='yeo_johnson', metadata={
        'help': 'sklearn numerical transformer to preprocess numerical data',
        'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']})
    mlp_division: int = field(default=4, metadata={
        'help': 'the ratio of the number of '
                'hidden dims in a current layer to the next MLP layer'})
    combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat', metadata={
        'help': 'method to combine categorical and numerical features, '
                'see README for all the method'})
    mlp_dropout: float = field(default=0.1, metadata={'help': 'dropout ratio used for MLP layers'})
    numerical_bn: bool = field(default=True, metadata={'help': 'whether to use batchnorm on numerical features'})
    use_simple_classifier: str = field(default=True, metadata={
        'help': 'whether to use single layer or MLP as final classifier'})
    mlp_act: str = field(default='relu', metadata={
        'help': 'the activation function to use for finetuning layers',
        'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']})
    gating_beta: float = field(default=0.2, metadata={
        'help': "the beta hyperparameters used for gating tabular data "
                "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"})

    # def __post_init__(self):
    #     assert self.column_info != self.column_info_path
    #     if self.column_info is None and self.column_info_path:
    #         with open(self.column_info_path, 'r') as f:
    #             self.column_info = json.load(f)
    # def to_json_string(self):
    #     """
    #     Serializes this instance to a JSON string.
    #     """
    #     return json.dumps(self.to_dict(), fp, indent=4, sort_keys=True)
