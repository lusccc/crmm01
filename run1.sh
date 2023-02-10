WANDB_PROJECT=crmm

#cuda='--no_cuda'
cuda=''
#task='pretrain_multi_modal_dbn'
task='fine_tune_multi_modal_dbn_classification'

#modality='num,cat,text'
#modality='cat,text'
modality='cat,num'

#text_cols="Name,Symbol,Rating Agecncy Name,Sector,secKeywords"
text_cols="Name,Symbol,Rating Agecncy Name,Sector,secKeywords"
#text_cols="secKeywords"
cat_cols='Symbol,Sector,CIK'

use_rbm_for_text='--use_rbm_for_text' # true
#use_rbm_for_text='' # false

#pretrained_multi_modal_dbn_model_dir='none'
pretrained_multi_modal_dbn_model_dir='exps/pretrain_multi_modal_dbn_2023-02-09_11-00-41_mSK/output'

fp16='--fp16'
per_device_train_batch_size=91
num_train_epochs=150
patience=100

python crmm/crmm_dbn_tabular_classification.py --modality ${modality} --task ${task} ${cuda} ${use_rbm_for_text}  --pretrained_multi_modal_dbn_model_dir ${pretrained_multi_modal_dbn_model_dir} ${fp16} --per_device_train_batch_size ${per_device_train_batch_size} --num_train_epochs ${num_train_epochs} --patience ${patience}

#echo "python crmm/crmm_dbn_tabular_classification.py --modality ${modality} --task ${task} ${cuda} ${use_rbm_for_text}  --pretrained_multi_modal_dbn_model_dir ${pretrained_multi_modal_dbn_model_dir} ${fp16} --per_device_train_batch_size ${per_device_train_batch_size} --num_train_epochs ${num_train_epochs} --patience ${patience}"
