# #!/bin/bash
export CUDA=1
export CV_NUM=0 # 5-fold cv {0,1,2,3,4}
export pretrain_model_type=BiomedNLP-PubMedBERT-base-uncased-abstract ## bert-base-uncased; BiomedNLP-PubMedBERT-base-uncased-abstract
export pretrain_model_dir=../data/pretrain_model

export train_dir=../data/prepro_target_data/trec-covid # also is target_dir
export save_dir=../results

export checkpoint_name=step_best  ## step_best or step_n
export checkpoint_folder=../results/train__meta_CTSyncSup_trec-covid_trec-covid/0602-2234-04__meta_CTSyncSup_trec-covid_trec-covid__fold_0
## meta-pretrained checkpoint

## ------------------------------------------------------------------
## ------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$CUDA python ../meta_scripts/train.py --run_mode train \
--cv_number $CV_NUM \
--mode_name metafine \
--train_dir $train_dir \
--target_dir $train_dir \
--save_dir $save_dir \
--pretrain_model_dir $pretrain_model_dir \
--pretrain_model_type $pretrain_model_type \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 256 \
--eval_during_train \
--save_checkpoint \
--load_checkpoint \
--checkpoint_name $checkpoint_name \
--load_checkpoint_folder $checkpoint_folder/checkpoints \