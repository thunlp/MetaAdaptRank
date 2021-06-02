import os
import sys
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import pytrec_eval


def add_default_args(parser):
    parser.add_argument('--result_fold_0', type=str, required=True,
                        help="The path to be combine sub results.")
    parser.add_argument('--result_fold_1', type=str, required=True,
                        help="The path to be combine sub results.")
    parser.add_argument('--result_fold_2', type=str, required=True,
                        help="The path to be combine sub results.")
    parser.add_argument('--result_fold_3', type=str, required=True,
                        help="The path to be combine sub results.")
    parser.add_argument('--result_fold_4', type=str, required=True,
                        help="The path to be combine sub results.")
    parser.add_argument('--qrel_path', type=str, required=True,
                        help="The path to the qrel file.")
    parser.add_argument('--combine_neuscore', action="store_true", default=False,
                        help="Whether combine bert eval scores.")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="The path to save combined features.")
    

def create_folder_fct(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        

def combine_trec(result_folders, save_filename, trec_name="best_test"):
    tot_num = 1
    with open(save_filename, mode="w", encoding="utf-8") as writer:
        for folder in result_folders:
            trec_path = os.path.join(folder, "%s.trec"%trec_name)
            with open(trec_path, "r", encoding="utf-8") as reader:
                for line in reader:
                    writer.write(line)
                    tot_num += 1
            reader.close()
    writer.close()
    print("success saved %d-line results to %s"%(tot_num, save_filename))
    

def compute_trec_metrics(trec_path, qrels_path):
    with open(trec_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
    
    with open(qrels_path, 'r') as f_qrel:
        qrels = pytrec_eval.parse_qrel(f_qrel)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)
    for query_id, query_measures in sorted(results.items()):
        pass
    mes = {}
    for measure in sorted(query_measures.keys()):
        mes[measure] = pytrec_eval.compute_aggregated_measure(
            measure, [query_measures[measure] for query_measures in results.values()]
        )
    return mes


def load_txt_to_list(file_path, spliter=None, terminator="\n"):
    """ Load file.txt by lines, convert it to list. """
    txt_list = []
    with open(file=file_path, mode='r', encoding='utf-8') as fi:
        for i, line in enumerate(tqdm(fi)):
            if spliter is None:
                line = line.strip(terminator)
            else:
                line = line.strip(terminator).split(spliter)
            txt_list.append(line)
    return txt_list


def load_qrels_to_map(qrel_path, qrel_spliter=" "):
    # load qrels
    qrels = load_txt_to_list(
        file_path=qrel_path, 
        spliter=qrel_spliter
    )
    # convert qrel to map
    qid2item = {}
    for qrel in qrels:
        assert len(qrel) == 4
        qid, _, doc_id, label = qrel
        label = int(label)
        if qid not in qid2item:
            qid2item[qid] = {doc_id:label}
        else:
            qid2item[qid][doc_id] = label
    return qid2item


def assert_qids(features_qids, qrels_qids):
    for qid in features_qids:
        assert qid in qrels_qids
    if len(features_qids) != len(qrels_qids):
        print("features_qids_num = {} | qrels_qids_num = {}".format(len(features_qids), len(qrels_qids)))
    assert len(features_qids) == len(qrels_qids)


def sort_list(data_list, index, reverse):
    """ index: int number, according to it to sort data_list"""
    if index is None:
        sorted_data_list = sorted(data_list, key=lambda x: x, reverse=reverse)
    else:
        sorted_data_list = sorted(data_list, key=lambda x: x[index], reverse=reverse)
    return sorted_data_list


def combine_feature_files(result_folders, combine_neuscore, feature_name="best_test_features"):
    qid2docid_features = {}
    
    fold_list = [[folder, int(folder.split("_")[-1])] for folder in result_folders]
    sort_fold_list = sort_list(fold_list, index=1, reverse=False)
    tot_num = 0
    for folder, _ in sort_fold_list:
        feature_path = os.path.join(folder, "%s.jsonl"%feature_name)
        with open(feature_path, "r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                qid = data["qid"]
                docid = data["docid"]
                
                feature = []
                for fea in data["neural_feature"]:
                    feature.append(fea)
                    
                if combine_neuscore:
                    feature.append(float(data["neural_score"]))
                    
                feature.append(float(data["retrival_score"]))
                tot_num += 1

                if qid not in qid2docid_features:
                    qid2docid_features[qid] = {docid:feature}
                else:
                    if docid not in qid2docid_features[qid]:
                        qid2docid_features[qid][docid] = feature
                    else:
                        print("Error! duplicate qid2docid")
    print("success combine %d-line features"%tot_num)   
    return qid2docid_features

    
    
def gen_ranklib_features(features, labels, norecord_label=0):
    feature_list = []
    no_record_num = 0
    for qid in features:
        for docid in features[qid]:
            feature_line = []
            # --------------------------------
            # label
            if docid in labels[qid]:
                label = labels[qid][docid]
            else:
                label = norecord_label
                no_record_num += 1
            feature_line.append(str(label))
            # --------------------------------
            # qid
            feature_line.append('qid:%s'%qid)
            # --------------------------------
            # feature
            for idx, fea in enumerate(features[qid][docid]):
                feature_line.append("%s:%s"%(str(idx+1), str(fea)))
            # --------------------------------
            # # docid
            feature_line.append("#%s"%docid)
        # --------------------------------
            feature_list.append(" ".join(feature_line))
            
    print("no_record_num = %d"%no_record_num)
    return feature_list


def save_combined_features(ranklib_features, save_filename):
    with open(save_filename, 'w', encoding="utf-8") as writer:
        for feature in ranklib_features:
            writer.write("{}\n".format(feature))
    writer.close()
    print("success saved ranklib features to %s !"%save_filename)
    

if __name__ == "__main__":
    # setting args
    parser = argparse.ArgumentParser(
        'FeatureCombiner', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_default_args(parser)
    args = parser.parse_args()
    
    # create save folder
    create_folder_fct(args.save_dir)
    
    ## **********************
    # combine trec files
    ## **********************
    combine_trec_filename = os.path.join(args.save_dir, "%s.trec"%args.save_dir.split("/")[-1])
    
    tot_trec = combine_trec(
        result_folders=[args.result_fold_0, args.result_fold_1, args.result_fold_2, args.result_fold_3, args.result_fold_4],
        save_filename=combine_trec_filename
    )
    metrics = compute_trec_metrics(trec_path=combine_trec_filename, qrels_path=args.qrel_path)
    print("[Combine results] NDCG@20 = %.4f | P_20 = %.4f "%(metrics["ndcg_cut_20"], metrics["P_20"]))
    
    ## **********************
    # combine features files
    ## **********************
    combine_features_filename = os.path.join(args.save_dir, "features.txt")
    
    # load qrels
    label_qid2docids = load_qrels_to_map(args.qrel_path)
    
    # load features
    features = combine_feature_files(
        result_folders=[args.result_fold_0, args.result_fold_1, args.result_fold_2, args.result_fold_3, args.result_fold_4],
        combine_neuscore=args.combine_neuscore
    )
    assert_qids(features.keys(), label_qid2docids.keys())
    
    # convert to ranklib format
    ranklib_features = gen_ranklib_features(features, label_qid2docids)
    
    # save combined ranklin features
    save_combined_features(ranklib_features, save_filename=combine_features_filename)