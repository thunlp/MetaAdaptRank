export trec_eval=./utils/trec_eval.9.0.4/trec_eval
export gdeval=./utils/gdeval.pl

export cv_num=5
export metric_name=NDCG@20

# **********************************************************
export qrel_path=../data/prepro_target_data/clueweb09/fold_0/qrels # qrels of all folds are the same
export feature_name=metafine_bert_cw09
export ranklib_path=../combined_features/$feature_name
# **********************************************************

# [1] split features to cv_num folds
java -cp ./utils/RankLib-2.1-patched.jar ciir.umass.edu.features.FeatureManager \
-input $ranklib_path/features.txt \
-output $ranklib_path/ \
-k $cv_num

# [2] train cv_num models to ca
java -jar ./utils/RankLib-2.1-patched.jar \
-train $ranklib_path/features.txt \
-ranker 4 \
-kcv $cv_num \
-kcvmd $ranklib_path/ \
-kcvmn ca \
-metric2t $metric_name \
-metric2T $metric_name

# [3] generate ca to score file
java -jar ./utils/RankLib-2.1-patched.jar -load $ranklib_path/f1.ca -rank $ranklib_path/f1.test.features.txt -score $ranklib_path/f1.score
java -jar ./utils/RankLib-2.1-patched.jar -load $ranklib_path/f2.ca -rank $ranklib_path/f2.test.features.txt -score $ranklib_path/f2.score
java -jar ./utils/RankLib-2.1-patched.jar -load $ranklib_path/f3.ca -rank $ranklib_path/f3.test.features.txt -score $ranklib_path/f3.score
java -jar ./utils/RankLib-2.1-patched.jar -load $ranklib_path/f4.ca -rank $ranklib_path/f4.test.features.txt -score $ranklib_path/f4.score
java -jar ./utils/RankLib-2.1-patched.jar -load $ranklib_path/f5.ca -rank $ranklib_path/f5.test.features.txt -score $ranklib_path/f5.score

# [4] combine split scores to trec format
python ./utils/combine_ranklib_results.py \
--cv_num $cv_num \
--ranklib_path $ranklib_path \

## **************************************************
$gdeval -c $qrel_path $ranklib_path/coor_ascent.$feature_name.trec
$trec_eval -m ndcg_cut.20 $qrel_path $ranklib_path/coor_ascent.$feature_name.trec
$trec_eval -m P.20 $qrel_path $ranklib_path/coor_ascent.$feature_name.trec