#!/bin/bash
set -e
create_exp_dirs() {
  root_dir="$1"
  task="$2"

  if [ ! -d "$root_dir" ]; then
    echo "Root directory '$root_dir', where the directory of the experiment will be created, must exist"
    exit 1
  fi

  formatted_timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
  output_dir="$root_dir/$task"
  rand_suffix=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 3 | head -n 1)
  output_dir="$output_dir"_"$formatted_timestamp"_"$rand_suffix"
  exp_args_output_dir="$output_dir/output"
  exp_args_logging_dir="$output_dir/logging"

  mkdir -p "$exp_args_output_dir" "$exp_args_logging_dir"

  # 输出 exp_args_output_dir 和 exp_args_logging_dir 变量的值，以便调用函数后可以捕获这些值
  echo "$exp_args_output_dir"
  echo "$exp_args_logging_dir"
}

run_single_experiment() {
  root_dir=$1
  task=$2
  bert_model=$3
  modality=$4
  batch_size=$5
  pre_epoch=$6
  finetune_epoch=$7
  patience=$8
  dataset_name=$9
  data_path=${10}
  excel_path=${11}
  scratch=${12}

  if [ "$scratch" = "yes" ]; then
    task=fine_tune_from_scratch
    exp_dirs=($(create_exp_dirs $root_dir $task))
    output_dir=${exp_dirs[0]}
    logging_dir=${exp_dirs[1]}
    python main_runner.py --task $task --bert_model_name $bert_model --use_modality $modality --per_device_train_batch_size $batch_size --num_train_epochs $finetune_epoch --patience $patience --dataset_name $dataset_name --data_path $data_path --output_dir $output_dir --logging_dir $logging_dir --save_excel_path $excel_path
  else
    task=pretrain
    pretrain_exp_dirs=($(create_exp_dirs $root_dir $task))
    pretrain_output_dir=${pretrain_exp_dirs[0]}
    pretrain_logging_dir=${pretrain_exp_dirs[1]}
    python main_runner.py --task pretrain --bert_model_name $bert_model --use_modality $modality --per_device_train_batch_size $batch_size --num_train_epochs $pre_epoch --patience $patience --dataset_name $dataset_name --data_path $data_path --output_dir $pretrain_output_dir --logging_dir $pretrain_logging_dir --save_excel_path $excel_path

    task=fine_tune
    fine_tune_exp_dirs=($(create_exp_dirs $root_dir $task))
    fine_tune_output_dir=${fine_tune_exp_dirs[0]}
    fine_tune_logging_dir=${fine_tune_exp_dirs[1]}
    python main_runner.py --task fine_tune --bert_model_name $bert_model --use_modality $modality --per_device_train_batch_size $batch_size --num_train_epochs $finetune_epoch --patience $patience --dataset_name $dataset_name --data_path $data_path --output_dir $fine_tune_output_dir --logging_dir $fine_tune_logging_dir --pretrained_model_dir $pretrain_output_dir --save_excel_path $excel_path
  fi
}

generate_dataset() {
  dataset_name=$1
  sent_num=$2
  kw_num=$3

  python crmm/data_acquisition/5sec_text_sumy.py --sentences_num $sent_num --dataset_name $dataset_name
  python crmm/data_acquisition/6sec_keywords_extraction.py --prev_step_sentences_num $sent_num --keywords_num $kw_num --dataset_name $dataset_name

}

split_dataset() {
  dataset_name=$1
  sent_num=$2
  kw_num=$3
  n_class=$4
  split_method=$5
  train_years=$6
  test_years=$7
  echo "python crmm/data_acquisition/7.2dataset_split.py --dataset_name $dataset_name --prev_step_sentences_num $sent_num --prev_step_keywords_num $kw_num --n_class $n_class --split_method $split_method --train_years $train_years --test_years $test_years"
  python crmm/data_acquisition/7.2dataset_split.py --dataset_name $dataset_name --prev_step_sentences_num $sent_num --prev_step_keywords_num $kw_num --n_class $n_class --split_method $split_method --train_years $train_years --test_years $test_years
}

# @@@@@@@@ EXPS: @@@@@@@
# mixed split method exps

loop_pre_epoch() {
  root_dir=$1
  task=$2
  bert_model=$3
  modality=$4
  batch_size=$5
  pre_epoch=$6
  finetune_epoch=$7
  patience=$8
  dataset_name=$9
  data_path=${10}
  excel_path=${11}
  pre_epochs=(3 5 10 15 20 25 30 40 50)
  scratch=no

  for epoch in "${pre_epochs[@]}"; do
    run_single_experiment $root_dir $task $bert_model $modality $batch_size $epoch $finetune_epoch $patience $dataset_name $data_path $excel_path $scratch
  done
}

loop_modality_and_scratch() {
  for modality in "${modalities[@]}"; do
    for scratch in "${scratch_values[@]}"; do
      echo "Running experiment with bert_model: $bert_model, modality: $modality, scratch: $scratch"
      run_single_experiment $scratch $modality $pre_epoch
    done
  done
}

# @@@@@ RUN @@@@@
declare -A params=(
    ["dataset_name"]="cr"
    ["sent_num"]=10
    ["kw_num"]=20
    ["n_class"]=2
    ["split_method"]="mixed"
    ["train_years"]="-1"
    ["test_years"]="-1"
)
generate_dataset $params
split_dataset $params

declare -A bert_model_batch_sizes=(
  ["bert-base-uncased"]=100
  ["nickmuchi/sec-bert-finetuned-finance-classification"]=100
  ["ProsusAI/finbert"]=100
  ["prajjwal1/bert-tiny"]=1000
)
bert_model="prajjwal1/bert-tiny"
bert_model_name='bert_tiny'
batch_size=${bert_model_batch_sizes[$bert_model]}
pre_epoch=15
finetune_epoch=200
patience=1000
n_class=2
dataset_name="cr2_sec_${n_class}"
data_path="./data/cr2_sec_${n_class}"
root_dir='./exps'
excel_path="excel/cls${n_class}_${bert_model_name}_sent10_kw20.xlsx"

modalities=('num' 'cat' 'text' 'num,cat' 'num,text' 'cat,text' 'num,cat,text')
scratch_values=('yes' 'no')
loop_modality_scratch
