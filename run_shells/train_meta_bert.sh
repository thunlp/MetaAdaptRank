# #!/bin/bash
export CUDA=0
export CV_NUM=0 # 5-fold cv {0,1,2,3,4}
export pretrain_model_type=BiomedNLP-PubMedBERT-base-uncased-abstract ## bert-base-uncased; BiomedNLP-PubMedBERT-base-uncased-abstract

export pretrain_model_dir=../data/pretrain_model 
export train_dir=../data/synthetic_data/CTSyncSup_trec-covid
export target_dir=../data/prepro_target_data/trec-covid
export save_dir=../results

## ------------------------------------------------------------------
## ------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$CUDA python ../meta_scripts/train.py --run_mode train \
--cv_number $CV_NUM \
--mode_name meta \
--train_dir $train_dir \
--target_dir $target_dir \
--save_dir $save_dir \
--pretrain_model_dir $pretrain_model_dir \
--pretrain_model_type $pretrain_model_type \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 256 \
--eval_during_train \
--save_checkpoint \