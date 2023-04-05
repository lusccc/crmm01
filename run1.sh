WANDB_PROJECT=crmm


no_cuda='false'
task='fine_tune_multi_modal_dbn_classification_scratch'

#modalities='num,cat,text'
#modalities='cat,text'
# modalities='cat,num'
#modalities='num'
modality='text'



#pretrained_model_dir='none'
pretrained_model_dir='exps/pretrain_multi_modal_dbn_2023-03-13_14-05-48_QDW/output'

if [ "$task" = "pretrain" ]; then
  per_device_train_batch_size=1000
elif [ "$task" = "fine_tune" ]; then
  per_device_train_batch_size=1000
elif [ "$task" = "fine_tune_from_scratch" ]; then
  per_device_train_batch_size=1000
else
  per_device_train_batch_size=1000
fi

fp16='true'
#fp16='false'
num_train_epochs=150
patience=100

python crmm/crmm_dbn_tabular_classification.py --modality ${modality} --task ${task} --no_cuda ${no_cuda} --use_rbm_for_text ${use_rbm_for_text}  --pretrained_multi_modal_dbn_model_dir ${pretrained_multi_modal_dbn_model_dir} --fp16 ${fp16} --per_device_train_batch_size ${per_device_train_batch_size} --num_train_epochs ${num_train_epochs} --patience ${patience} --numerical_transformer_method yeo_johnson --dataloader_num_workers 20

#echo "python crmm/crmm_dbn_tabular_classification.py --modalities ${modalities} --task ${task} ${cuda} ${use_rbm_for_text}  --pretrained_multi_modal_dbn_model_dir ${pretrained_multi_modal_dbn_model_dir} ${fp16} --per_device_train_batch_size ${per_device_train_batch_size} --num_train_epochs ${num_train_epochs} --patience ${patience}"