# !/bin/bash
## --------------------------------------------
export CUDA=2
export generator_mode=contrastqg ## qg; contrastqg
export pretrain_generator_type=t5-small ## t5-small ; t5-base
export target_dataset_name=trec-covid ## clueweb09; robust04; trec-covid
export per_gpu_gen_batch_size=200
## --------------------------------------------

## --------------------------------------------
# export generator_load_dir=../results/train__qg/qg-t5-small/checkpoints
export generator_load_dir=../results/train__contrastqg/contrastqg-t5-small/checkpoints
## --------------------------------------------

CUDA_VISIBLE_DEVICES=$CUDA python ../nlg_scripts/inference.py --run_mode inference \
--generator_mode $generator_mode \
--pretrain_generator_type $pretrain_generator_type \
--per_gpu_gen_batch_size $per_gpu_gen_batch_size \
--generator_load_dir $generator_load_dir \
--target_dataset_dir ../data/prepro_target_data/$target_dataset_name \
--save_dir ../results \