from transformers import HfArgumentParser, TrainingArguments

from arguments import CrmmTrainingArguments

parser = HfArgumentParser([CrmmTrainingArguments])
parser.parse_args_into_dataclasses()