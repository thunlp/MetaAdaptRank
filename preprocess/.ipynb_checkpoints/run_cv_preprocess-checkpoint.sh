# #!/bin/bash
## ****************************
export dataset_name=trec-covid ## clueweb09; robust04; trec-covid
## ****************************

python ./utils/cv_spliter.py \
--dataset_name $dataset_name \
--input_path ../data/target_data/$dataset_name \
--output_path ../data/prepro_target_data/$dataset_name