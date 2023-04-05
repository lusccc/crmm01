from dataclasses import dataclass, field
from typing import Optional, Union, List

from transformers import TrainingArguments, IntervalStrategy

TASK = ['pretrain', 'fine_tune', 'fine_tune_from_scratch']


@dataclass
class CrmmTrainingArguments(TrainingArguments):
    # !!!!OURS!!!!
    report_to: str = field(default='wandb')
    root_dir: str = field(default='./exps', metadata={"help": "parent dir of output_dir"})
    task: str = field(default="pretrain", metadata={"help": "", 'choices': TASK})
    use_modality: str = field(default="num,cat,text", metadata={"help": "used in multi dbn"})
    pretrained_model_dir: str = field(default='exps/pretrain_2023-04-02_09-08-31_YFE/output', metadata={"help": ""})
    patience: int = field(default=1000, metadata={"help": ""})

    # @@@@@@@@@@ BELOW ARE HUGGINGFACE ARGS
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
    # remove_unused_columns should be False for models.multi_modal_dbn.MultiModalDBN.forward
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    # @@@@@@@@@@ num_train_epochs
    num_train_epochs: float = field(default=1, metadata={"help": "Total number of training epochs to perform."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    # @@@@@@@@@ per_device_train_batch_size, 7 for dbn train
    per_device_train_batch_size: int = field(default=300, metadata={
        "help": "Batch size per GPU/TPU core/CPU for training."})
    evaluation_strategy: Union[IntervalStrategy, str] = field(default="epoch", metadata={
        "help": "The evaluation strategy to use."}, )  # metric will be checked in early-stopping
    dataloader_num_workers: int = field(default=16, metadata={
        "help": ("Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                 " in the main process.")})
    save_strategy: Union[IntervalStrategy, str] = field(default="epoch", metadata={
        "help": "The checkpoint save strategy to use."})
    load_best_model_at_end: Optional[bool] = field(default=True, metadata={
        "help": "Whether or not to load the best model found during training at the end of training."})
    save_total_limit: Optional[int] = field(default=1, metadata={
        "help": ("Limit the total amount of checkpoints. "
                 "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints")})
    auto_find_batch_size: bool = field(default=True, metadata={
        "help": ("Whether to automatically decrease the batch size in half and rerun the training loop again each time"
                 " a CUDA Out-of-Memory was reached")})
    # metric_for_best_model: str = field(default='loss')  # used for early stopping;
    greater_is_better: bool = field(default=False)  # used for early stopping
    fp16: bool = field(
        default=not no_cuda,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )

    # PyTorch 2.0 specifics
    # bf16: bool = field(default=True, metadata={})
    # torch_compile: bool = field(default=False, metadata={})
    # optim: str = field(default='adamw_torch_fused', metadata={})

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.use_modality, str):
            self.use_modality = [m.strip() for m in self.use_modality.split(',')]


@dataclass
class BertModelArguments:
    # prajjwal1/bert-tiny bert-base-uncased
    bert_model_name: str = field(default='prajjwal1/bert-tiny', metadata={
        "help": "Path to pretrained model or model identifier from huggingface.co/models"})
    max_seq_length: int = field(default=512, metadata={
        "help": "The maximum length (in number of tokens) for the inputs to the transformer model"})
    config_name: Optional[str] = field(default=None, metadata={
        "help": "Pretrained config name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default='./exps/model_config_cache', metadata={
        "help": "Where do you want to store the pretrained models downloaded from s3"})
    load_hf_model_from_cache: bool = field(default=True, )


@dataclass
class MultimodalDataArguments:
    dataset_name: str = field(default='cr_sec_6', metadata={"help": ""})
    data_path: str = field(default=f'./data/cr_sec_6', metadata={
        'help': 'the path to the csv file containing the dataset'})
    use_val: bool = field(default=True)
    column_info: dict = field(default=None, metadata={
        'help': 'a dict referencing the text, categorical, numerical, and label columns'
                'its keys are text_cols, num_cols, cat_cols, and label_col'})
    numerical_transformer_method: str = field(default='yeo_johnson', metadata={
        'help': 'sklearn numerical transformer to preprocess numerical data',
        'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'standard', 'none']})
