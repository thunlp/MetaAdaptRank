import os
import re
import json
import logging
from tqdm import tqdm
        
logger = logging.getLogger()

    
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# load qid2query & docid2doc
def load_corpus(
    args, 
    data_dir, 
    tokenizer,
    mode,
):
    """
    :param args: args parameters
    :param data_dir: qid2query, docid2doc
    :param mode: train, target, dev, test
    :param tokenizer: BertTokenizer or None
    """
    curr_dir = os.path.join(data_dir, 
                            "%s_tokenized"%args.pretrain_model_type)
    upper_dir = os.path.join("/".join(data_dir.split("/")[:-1]), 
                             "%s_tokenized"%args.pretrain_model_type)
    
    # cached tokenized qid2query & docid2doc
    if mode in ["target", "dev", "test"] or os.path.exists(upper_dir):
        cache_dir = upper_dir
    else:
        cache_dir = curr_dir
        
    # reload tokenized qid2query & docid2doc
    try:
        corpus = load_id2text(cache_dir)
        logger.info('success load tokenized corpus from %s !'%cache_dir)
    except:
        # load qid2query & docid2doc
        logger.info('start load corpus ...')
        orig_corpus = load_id2text(data_dir)
        # tokenize qid2query & docid2doc
        logger.info('convert text to toks...')
        corpus = convert_text2toks(
            orig_corpus, 
            tokenizer=tokenizer, 
            max_query_length=args.max_query_length,
            max_doc_length=args.max_doc_length, 
            do_lower_case=args.do_lower_case
        )
        # save cached tokenized dataset
        save_tokenized_corpus(corpus, cache_dir=cache_dir)
        logger.info('success saved cached tokenized corpus!')
            
    return corpus


def load_id2text(folder_path):
    """
    :param folder_path: data folder path
    :return: qid2query, docid2doc
    """
    dataset_dict = {}
    dataset_dict["qid2query"] = load_json2dict(
        os.path.join(folder_path, "qid2query.jsonl"), 
        id_name="qid", 
        text_key="query",
    ) # qid2query
    
    dataset_dict["docid2doc"] = load_json2dict(
        os.path.join(folder_path, "docid2doc.jsonl"), 
        id_name="docid", 
        text_key="doc",
    ) # docid2doc
            
    for key in dataset_dict:
        logger.info('success load %s = %d'
                    %(key, len(dataset_dict[key])))
    return dataset_dict


def load_json2dict(file_path, id_name, text_key):
    """used in load_dataset."""
    data_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as fi:
        for idx, line in enumerate(tqdm(fi)):
            data = json.loads(line)
            data_dict[data[id_name]] = data[text_key]
    return data_dict

def load_json2list(file_path):
    """used in load_dataset."""
    data_list = []
    with open(file_path, mode='r', encoding='utf-8') as fi:
        for idx, line in enumerate(tqdm(fi)):
            data = json.loads(line)
            data_list.append(data)
    return data_list


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def convert_text2toks(
    dataset, 
    tokenizer, 
    max_query_length, 
    max_doc_length, 
    do_lower_case
):
    """
    :param dataset: has key = qid2query / docid2doc
    :param tokenizer: BertTokenizer
    :param max_query_length: max query word length
    :param max_doc_length: max query doc length
    :param do_lower_case: Weather do lower case
    :return qid2query, docid2doc: tokenized tokens
    """
    max_len_dict = {
        "qid2query":max_query_length, 
        "docid2doc":max_doc_length
}
    for key in dataset:
        max_seq_len = max_len_dict[key]
        for ids in tqdm(dataset[key]):
            text = dataset[key][ids]
            # split text to tokens
            tokens = tokenizer.tokenize(text, max_seq_len)
            # update dict
            dataset[key][ids] = tokens
    return dataset


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def save_tokenized_corpus(dataset, cache_dir):
    """
    :param: dataset dict has keys : qid2query, docid2doc
    :param: save dir
    """
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
        
    save_dict2jsonl(
        data_dict=dataset["qid2query"], 
        output_path=os.path.join(cache_dir, "qid2query.jsonl"), 
        id_name="qid", 
        text_key="query"
    )
    save_dict2jsonl(
        data_dict=dataset["docid2doc"], 
        output_path=os.path.join(cache_dir, "docid2doc.jsonl"), 
        id_name="docid", 
        text_key="doc"
    )
    
def save_list2jsonl(data_list, output_path):
    with open(file=output_path, mode="w", encoding="utf-8") as fw:
        for data in data_list:
            fw.write("{}\n".format(json.dumps(data)))
        fw.close()
        
def save_dict2jsonl(data_dict, output_path, id_name, text_key):
    with open(file=output_path, mode="w", encoding="utf-8") as fw:
        for key in data_dict:
            data = {id_name:key, text_key:data_dict[key]}
            fw.write("{}\n".format(json.dumps(data)))
        fw.close()