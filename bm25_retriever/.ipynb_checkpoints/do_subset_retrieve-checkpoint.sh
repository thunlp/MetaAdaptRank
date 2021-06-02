export dataset_name=trec-covid ## clueweb09; robust04; trec-covid
export generator_folder=qg_t5-small ## qg_t5-small ; qg_t5-base
export data_path=../data/prepro_target_data/$dataset_name

## build index
./bin/IndexCollection -collection JsonCollection -input $data_path/bm25_corpus -index $dataset_name -generator LuceneDocumentGenerator -threads 8 -storePositions -storeDocvectors -storeRawDocs

## retrieve
./bin/SearchCollection -index $dataset_name -topicreader TsvString -topics $data_path/$generator_folder/qid2query.tsv -bm25 -hits 100 -output $data_path/$generator_folder/bm25_retrieval.trec