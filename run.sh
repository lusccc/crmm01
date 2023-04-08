modality='num,cat,text'
#modality='num,cat'
#modality='cat'
#modality='num'
#modality='text'
scratch='no'
#scratch='yes'

pre_epoch=5
finetune_epoch=300
patience=1000

bert_model='prajjwal1/bert-tiny'
#bert_model='bert-base-uncased'
#bert_model='nickmuchi/sec-bert-finetuned-finance-classification'
#bert_model='ProsusAI/finbert'

if [ "$bert_model" = "bert-base-uncased" ] || [ "$bert_model" = "nickmuchi/sec-bert-finetuned-finance-classification" ] || [ "$bert_model" = "ProsusAI/finbert" ]
then
  batch_size=100
elif [ "$bert_model" = "prajjwal1/bert-tiny" ]; then
  batch_size=300
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
