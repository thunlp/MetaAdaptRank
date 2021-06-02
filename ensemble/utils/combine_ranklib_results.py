import os
import sys
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import pytrec_eval


def add_default_args(parser):
    parser.add_argument('--cv_num', type=int, default=5,
                        help="The number of cross-validation.")
    parser.add_argument('--ranklib_path', type=str, required=True,
                        help="The path of coor-ascent results")

def save_trec_result(rst_dict, save_filename, trec_mark="coor_ascent"):
    # save best
    with open(save_filename, "w", encoding="utf-8") as fo:
        for qid, item in rst_dict.items():
            res = sorted(item, key=lambda x: x[1], reverse=True)
            rank = 1
            docid_set = set()
            for docid, d_score in res:
                if docid in docid_set:
                    continue
                fo.write(" ".join([qid, "Q0", str(docid), str(rank), str(d_score), trec_mark]) + "\n")
                docid_set.add(docid)
                rank += 1
        fo.close()
        

def combine_coor_results(ranklib_path, cv_num):
    rst_dict = {}
    for i in tqdm(range(cv_num)):
        fold_i = i + 1
        score_path = os.path.join(ranklib_path, "f%d.score"%fold_i)
        feature_path = os.path.join(ranklib_path, "f%d.test.features.txt"%fold_i)
        with open(score_path, "r", encoding="utf-8") as scorer, \
        open(feature_path, "r", encoding="utf-8") as featurer:
            for score_line, feature_line in zip(scorer, featurer):
                score_data = score_line.strip("\n").split("\t")
                feature_data = feature_line.strip("\n").split(" ")
                # check qid
                qid = score_data[0]
                feature_qid = feature_data[1].split(":")[-1]
                assert qid == feature_qid
                
                # docid & coor_score
                docid = feature_data[-1].strip("#")
                coor_score = score_data[-1]
                
                if qid not in rst_dict:
                    rst_dict[qid] = [(docid, coor_score)]
                else:
                    rst_dict[qid].append((docid, coor_score))
                    
    return rst_dict


def save_trec_result(rst_dict, save_filename, trec_mark="coor_ascent"):
    # save best
    with open(save_filename, "w", encoding="utf-8") as fo:
        for qid, item in rst_dict.items():
            res = sorted(item, key=lambda x: x[1], reverse=True)
            rank = 1
            docid_set = set()
            for docid, d_score in res:
                if docid in docid_set:
                    continue
                fo.write(" ".join([qid, "Q0", str(docid), str(rank), str(d_score), trec_mark]) + "\n")
                docid_set.add(docid)
                rank += 1
        fo.close()
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        'CoorScoreCombiner', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ) 
    add_default_args(parser)
    args = parser.parse_args()
    
    rst_dict = combine_coor_results(
        ranklib_path=args.ranklib_path,
        cv_num=args.cv_num
    )
    
    save_trec_result(rst_dict, save_filename=os.path.join(args.ranklib_path, "coor_ascent.%s.trec"%args.ranklib_path.split("/")[-1]))