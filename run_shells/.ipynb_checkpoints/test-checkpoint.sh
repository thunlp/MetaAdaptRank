# #!/bin/bash
export CUDA=3
export CV_NUM=0 # 5-fold cv {0,1,2,3,4}
export mode_name=metafine ## meta or metafine

export pretrain_model_type=BiomedNLP-PubMedBERT-base-uncased-abstract ## bert-base-uncased; BiomedNLP-PubMedBERT-base-uncased-abstract
export pretrain_model_dir=../data/pretrain_model
export target_dir=../data/prepro_target_data/trec-covid
export save_dir=../results

export checkpoint_name=step_best  ## step_best or step_n
export checkpoint_folder=../results/train__metafine_trec-covid/0602-2151-01__metafine_trec-covid__fold_0

## ------------------------------------------------------------------
## ------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$CUDA python ../meta_scripts/test.py --run_mode test \
--cv_number $CV_NUM \
--mode_name $mode_name \
--target_dir $target_dir \
--save_dir $save_dir \
--pretrain_model_dir $pretrain_model_dir \
--pretrain_model_type $pretrain_model_type \
--per_gpu_eval_batch_size 256 \
--load_checkpoint \
--checkpoint_name $checkpoint_name \
--load_checkpoint_folder $checkpoint_folder/checkpoints \