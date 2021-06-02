export trec_eval=./utils/trec_eval.9.0.4/trec_eval
export gdeval=./utils/gdeval.pl

## **************************************************
export qrel_path=../data/prepro_target_data/clueweb09/fold_0/qrels # qrels of all folds are the same

export result_folder=../results/test__metafine_clueweb09
export result_fold_0=$result_folder/0721-0511-42__metafine_bert__clueweb09__fold_0
export result_fold_1=$result_folder/0721-0507-44__metafine_bert__clueweb09__fold_1
export result_fold_2=$result_folder/0721-0509-30__metafine_bert__clueweb09__fold_2
export result_fold_3=$result_folder/0721-0510-33__metafine_bert__clueweb09__fold_3
export result_fold_4=$result_folder/0721-1735-26__metafine_bert__clueweb09__fold_4

export save_folder_name=metafine_bert_cw09
export save_dir=../combined_features/$save_folder_name

## **************************************************
## combine trec & feature files
python ./utils/combine_features.py --result_fold_0 $result_fold_0 \
--result_fold_1 $result_fold_1 \
--result_fold_2 $result_fold_2 \
--result_fold_3 $result_fold_3 \
--result_fold_4 $result_fold_4 \
--qrel_path $qrel_path \
--save_dir $save_dir \

## **************************************************
$gdeval -c $qrel_path $save_dir/$save_folder_name.trec
$trec_eval -m ndcg_cut.20 $qrel_path $save_dir/$save_folder_name.trec
$trec_eval -m P.20 $qrel_path $save_dir/$save_folder_name.trec