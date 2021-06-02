export dataset_name=trec-covid ## clueweb09; robust04; trec-covid

python ./utils/prepro_nlg_dataset.py \
--dataset_name $dataset_name \
--input_path ../data/target_data \
--output_path ../data/prepro_target_data