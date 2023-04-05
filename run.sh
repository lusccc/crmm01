#modality='num,cat,text'
modality='num,cat'
#modality='cat'
#modality='num'
#modality='text'
scratch='no'
#scratch='no'

batch_size=100
pre_epoch=100
finetune_epoch=400
patience=100

bert_model='prajjwal1/bert-tiny'
#bert_model='bert-base-uncased'

if [ "$bert_model" = "bert-base-uncased" ]; then
  batch_size=100
elif [ "$bert_model" = "prajjwal1/bert-tiny" ]; then
  batch_size=2000
else
  echo "Unknown bert_model value: $bert_model"
  exit 1
fi

if [ "$scratch" = "yes" ]; then
  python main_runner.py --task fine_tune_from_scratch --bert_model_name $bert_model --use_modality $modality --per_device_train_batch_size $batch_size --num_train_epochs $finetune_epoch --patience $patience
else
  python main_runner.py --task pretrain --bert_model_name $bert_model --use_modality $modality --per_device_train_batch_size $batch_size --num_train_epochs $pre_epoch --patience $patience

  pretrain_model_path=$(head -n 1 'pretrain_path_transit.txt')

  python main_runner.py --task fine_tune --bert_model_name $bert_model --use_modality $modality --per_device_train_batch_size $batch_size --num_train_epochs $finetune_epoch --pretrained_model_dir $pretrain_model_path --patience $patience
fi
