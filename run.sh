python src/crmm_dbn_tabular_classification.py --exp_name  num_ae_pretrain --task num_ae_unsuperv_pretrain
python src/crmm_dbn_tabular_classification.py --exp_name  cat_ae_pretrain --task cat_ae_unsuperv_pretrain

tensorboard --port=6006 --logdir .