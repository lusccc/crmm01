WANDB_PROJECT=crmm
#python crmm/crmm_dbn_tabular_classification.py --modality num,cat --no_cuda
python crmm/crmm_dbn_tabular_classification.py --modality num,cat,text --task pretrain_multi_modal_dbn

#python crmm/crmm_dbn_tabular_classification.py --modality num,cat --task fine_tune_multi_modal_dbn_classification_scratch --num_train_epochs 250