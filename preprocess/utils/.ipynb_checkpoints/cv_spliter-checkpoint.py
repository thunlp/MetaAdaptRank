import os
import sys
import json
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm



def add_default_args(parser):
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        choices=["clueweb09", "robust04", "trec-covid"],
        help="Preprocessing dataset."
    )
    parser.add_argument(
        '--cv_num', 
        type=int, 
        default=5,
        help="Cross validation number."
    )
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True,
        help="Input path of orignal dataset path."
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True,
        help="Output path of preprocessed dataset."
    )
    parser.add_argument(
        '--topk', 
        type=int, 
        default=100,
        help="Reranking depth."
    )
    
    
def create_folder_fct(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

## load qrels
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
#             txt_list.append([l for l in line if len(l) > 0])
    return txt_list


def load_qrels_to_map(qrel_path, qrel_spliter=" "):
    """
    convert qrels.txt to qid2doc2label dict.
    """
    qrels = load_txt_to_list(
        file_path=qrel_path, 
        spliter=qrel_spliter
    )
    qid2item = {}
    label_set = set()
    docid_set = set()
    for qrel in qrels:
        assert len(qrel) == 4
        qid, _, doc_id, label = qrel
        label = int(label)
        if label < 0:label=0
        label_set.add(label)
        docid_set.add(doc_id)
        
        if qid not in qid2item:
            qid2item[qid] = {doc_id:label}
        else:
            qid2item[qid][doc_id] = label
            
    label_list = sorted(list(label_set), reverse=True)
    return qid2item, docid_set, label_list


## load docid2doc
def load_jsonl(file_path):
    """ Load file.jsonl ."""
    data_list = []
    with open(file_path, mode='r', encoding='utf-8') as fi:
        for idx, line in enumerate(tqdm(fi)):
            jsonl = json.loads(line)
            data_list.append(jsonl)
    logger.info('success load %d data'%len(data_list))
    return data_list


def load_docid2doc(input_path, body_key):
    data_list = load_jsonl(input_path)
    
    data_dict = {}
    for data in data_list:
        docid = data["docid"]
        
        title = data["title"].strip()
        bodytext = data[body_key].strip()
        doc = " ".join([title, bodytext]).strip().lower()
        
        if docid not in data_dict:
            data_dict[docid] = doc
    return data_dict


## load base retrieval file.
def load_trec_to_list(input_file, valid_qids, all_docids, topk):
    """
    Convert base retrieval scores to qid2docids & qid2docid2scores.
    """
    qid2docids, qid2docid2score = {}, {}
    with open(input_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip('\n').split(' ')
            assert len(line) == 6
            qid, _, docid, rank, score, _ = line
            
            if int(rank) > topk:
                continue
            
            if qid not in valid_qids:
                continue
                
            if docid not in all_docids:
                continue
            
            # qid2did_score
            if qid not in qid2docids:
                qid2docids[qid] = set()
                qid2docids[qid].add(docid)
                qid2docid2score[qid] = {docid: score}
            else:
                qid2docids[qid].add(docid)
                qid2docid2score[qid][docid] = score
    return qid2docids, qid2docid2score


# convert to qid2label2docidset
def get_qid2doc_grades(qid2item, valid_qid2did, label_list):
    """
    qid2label2docid
    We reset label <=0 to label=0.
    """
    assert qid2item.keys() == valid_qid2did.keys()
    
    qid2grades = {}
    for qid in valid_qid2did:
        grades = {item:set() for item in label_list}
        valid_dids = valid_qid2did[qid]

        for did in valid_dids:
            # non this doc record in qrels
            if did not in qid2item[qid]:
                grades[label_list[-1]].add(did)
            else:
                label = qid2item[qid][did]
                grades[label].add(did)
        qid2grades[qid] = grades    
    return qid2grades



def get_covid_splited_qids(valid_qids, cv_num):
    each_fold_num = int(len(valid_qids) / cv_num)
    splited_qid_list = []
    for k in range(cv_num):
        splited_qids = []
        for qid in range(k*each_fold_num+1, (k+1) * each_fold_num+1):
            splited_qids.append(str(qid))
        assert len(set(splited_qids)) == each_fold_num
        splited_qid_list.append(splited_qids)
    return splited_qid_list


# split cross-validation
def cv_qid_spliter(qids, cv_num, dataset_name):
    if dataset_name in ["clueweb09", "robust04"]:
        ## follow Dai et al.
        qid_dict = {}
        qid_pairs = [(qid, int(qid)) for qid in qids]
        for pair in qid_pairs:
            key = pair[1]%cv_num
            if key not in qid_dict:
                qid_dict[key] = [pair[0]]
            else:
                qid_dict[key].append(pair[0])
        return qid_dict
                
    elif dataset_name == "trec-covid":
        splited_qid_list = get_covid_splited_qids(qids, cv_num)

        qid_dict = {}
        for idx in range(len(splited_qid_list)):
            test_qids = splited_qid_list[idx]
            dev_qids = splited_qid_list[idx-1]

            # check non intersection
            assert len(set(test_qids) & set(dev_qids)) == 0

            train_qids = [qid for qid in qids if qid not in (set(test_qids) | set(dev_qids))]
            assert (set(train_qids) | set(test_qids) | set(dev_qids)) == set(qids)

            qid_dict[idx] = {
                "train_qids":train_qids,
                "dev_qids":dev_qids, 
                "test_qids":test_qids,
            }
        return qid_dict
    raise ValueError('Invalid dataset_name: [%s]' %dataset_name)
    
    

def merge_query(qid_dict, drop_key):
    qid_list = []
    for key in qid_dict.keys():
        if key != drop_key:
            qid_list.extend(qid_dict[key])
    return qid_list

    
def load_jsonl(file_path):
    """ Load file.jsonl ."""
    data_list = []
    with open(file_path, mode='r', encoding='utf-8') as fi:
        for idx, line in enumerate(tqdm(fi)):
            jsonl = json.loads(line)
            data_list.append(jsonl)
    return data_list

def get_valid_qid2query(input_path, valid_qids):
    """
    Load and filter valid qid2query.
    """
    data_list = load_jsonl(input_path)
    
    data_dict = {}
    for data in data_list:
        qid = data["qid"]
        query = data["query"]
        if qid not in data_dict:
            data_dict[qid] = query
    
    valid_data_list = []
    for qid in valid_qids:
        valid_data_list.append({"qid":qid, "query":data_dict[qid]})
    return valid_data_list


def get_valid_docid2doc(all_docid2doc, valid_docids):
    """
    filter valid docid2doc.
    """
    valid_data_list = []
    for docid in valid_docids:
        valid_data_list.append({"docid":docid, "doc":all_docid2doc[docid]})
    return valid_data_list

    

# Get training instances (triple format)
def get_train_triples(qid2label, qid_list, label_list):
    
    def get_neg_docs(neg_labels, label2docs):
        neg_docs = set()
        for label in neg_labels:
            for doc in label2docs[label]:
                neg_docs.add(doc)
        assert len(neg_docs) > 0
        return neg_docs

    # start here.
    tot_triples = []
    for qid in qid_list:
        label2docs = qid2label[qid]
        
        valid_labels = [label for label in label_list if len(label2docs[label]) > 0]
        
        if len(valid_labels) < 2:
            assert valid_labels[0] == label_list[-1]
            continue
        
        for idx, pos_label in enumerate(valid_labels):
            if (idx + 1) == len(valid_labels):break
            
            # pos docs
            pos_docs = label2docs[pos_label]
            
            # neg docs
            neg_labels = valid_labels[idx+1:]
            neg_docs = get_neg_docs(neg_labels, label2docs)
            
            # triples
            for p_doc in pos_docs:
                for n_doc in neg_docs:
                    tot_triples.append({"qid":qid, "pos_docid":p_doc, "neg_docid":n_doc})
                    
    return tot_triples
    
    
# get evaluation format (for reranking)
def get_eval_format(valid_qids, qid2docids, qid2docid2score):
    data_list = []
    for qid in valid_qids:
        for docid in qid2docids[qid]:
            score = qid2docid2score[qid][docid]
            data_list.append({"qid":qid, "docid":docid, "score":score})
    return data_list


# save preprocess dataset.
def save_jsonl(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as fo:
        for data in data_list:
            fo.write("{}\n".format(json.dumps(data)))
        fo.close()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        'CV_Spliter', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )    
    add_default_args(parser)
    args = parser.parse_args()
    
    # create output folder
    create_folder_fct(args.output_path)
    
    ## input filename
    qrel_path = os.path.join(args.input_path, "qrels")
    qid2query_path = os.path.join(args.input_path, "qid2query.jsonl")
    docid2doc_path = os.path.join(args.input_path, "docid2doc.jsonl")
    trec_path = os.path.join(args.input_path, "base_retrieval.trec")

    # load qrels 
    label_qid2docids, label_docid_set, label_list = load_qrels_to_map(qrel_path=qrel_path)
    print("label types = {}".format(label_list))
    
    # load docid2doc
    all_docid2doc = load_docid2doc(
        input_path=docid2doc_path,
        body_key="bodytext" if args.dataset_name in ["clueweb09", "robust04"] else "abstract",
    )

    # load base retrieval.trec 
    valid_qid2docids, qid2docid2score = load_trec_to_list(
        trec_path, 
        valid_qids=label_qid2docids.keys(), 
        all_docids=all_docid2doc.keys(),
        topk=args.topk
    )
    
    # enumerate valid qids, docids
    valid_qids = list(valid_qid2docids.keys())
    valid_docids = set([docid for qid in valid_qid2docids for docid in valid_qid2docids[qid]])
    
    
    print("valid query number = {}".format(len(valid_qids)))
#     print("label doc number = {}".format(len(label_docid_set)))
    print("valid doc number = {}".format(len(valid_docids)))

    
    
    # convert to qid2label2docidset
    qid2grades = get_qid2doc_grades(
        qid2item=label_qid2docids, 
        valid_qid2did=valid_qid2docids, 
        label_list=label_list
    )
    
    # split cv
    cv_qid_dict = cv_qid_spliter(
        qids=valid_qids, 
        cv_num=args.cv_num,
        dataset_name=args.dataset_name,
    )
    
    
    # get valid list{listqid2query, docid2doc}
    valid_qid2query = get_valid_qid2query(
        input_path=qid2query_path, 
        valid_qids=valid_qids
    )

    valid_docid2doc = get_valid_docid2doc(
        all_docid2doc=all_docid2doc,
        valid_docids=valid_docids
    )
    

    # save cv data
    for tar_key in range(args.cv_num):
        
        fold_path = os.path.join(args.output_path, 'fold_%d'%tar_key)
        create_folder_fct(fold_path)
        
        if args.dataset_name in ["clueweb09", "robust04"]:
            train_query = merge_query(cv_qid_dict, drop_key=tar_key)
            dev_query = train_query
            test_query = cv_qid_dict[tar_key]
            assert len(set(train_query) & set(test_query)) == 0
        else:
            train_query = cv_qid_dict[tar_key]["train_qids"]
            dev_query = cv_qid_dict[tar_key]["dev_qids"]
            test_query = cv_qid_dict[tar_key]["test_qids"]
            assert len(set(train_query) & set(dev_query) & set(test_query)) == 0
            
        print("Fold = {}| Train Query Num = {} | Dev Query Num = {} | Test Query Num = {}".format(
            tar_key, len(train_query), len(dev_query), len(test_query)))
        
        # train (triples)
        train_data = get_train_triples(
            qid2label=qid2grades, 
            qid_list=train_query, 
            label_list=label_list,
        )

        # dev (eval format) is the same as train
        dev_data = get_eval_format(
            valid_qids=dev_query, 
            qid2docids=valid_qid2docids, 
            qid2docid2score=qid2docid2score
        )
        
        # test (eval format)
        test_data = get_eval_format(
            valid_qids=test_query, 
            qid2docids=valid_qid2docids,
            qid2docid2score=qid2docid2score
        )
        
        ## *********************
        # save data
        ## *********************
        
        # qrels
        shutil.copyfile(qrel_path, os.path.join(fold_path, "qrels"))
        
        # qid2query
        save_jsonl(
            data_list=valid_qid2query, 
            filename=os.path.join(fold_path, "qid2query.jsonl")
        )
        
        # docid2doc
        save_jsonl(
            data_list=valid_docid2doc, 
            filename=os.path.join(fold_path, "docid2doc.jsonl")
        )

        # train
        save_jsonl(
            data_list=train_data, 
            filename=os.path.join(fold_path, "train.jsonl")
        )
        
        # dev
        save_jsonl(
            data_list=dev_data, 
            filename=os.path.join(fold_path, "dev.jsonl")
        )
        
        # test
        save_jsonl(
            data_list=test_data, 
            filename=os.path.join(fold_path, "test.jsonl")
        )
