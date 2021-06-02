export dataset_name=trec-covid ## clueweb09; robust04; trec-covid
export generator_folder=qg_t5-small ## qg_t5-small ; qg_t5-base

python ./utils/sample_contrast_pairs.py \
--dataset_name $dataset_name \
--generator_folder $generator_folder \
--input_path ../data/prepro_target_data \
--topk 100 \
--sample_n 5