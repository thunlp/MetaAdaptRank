def select_tokenizer(args):
    return BertTokenizer(args)

    
def select_data_loader(args):
    dataloder_dict = {"build_eval_dataset":eval_dataset}
    
    if args.loss_class == "pointwise":
        dataloder_dict["build_train_target_dataset"] = pair_dataset
        dataloder_dict["train_target_batchify"] = bert_pair_batchify_for_train
        dataloder_dict["eval_batchify"] = bert_batchify_for_eval
        return dataloder_dict
    
    elif args.loss_class == "pairwise":
        dataloder_dict["build_train_target_dataset"] = triple_dataset
        dataloder_dict["train_target_batchify"] = bert_triple_batchify_for_train
        dataloder_dict["eval_batchify"] = bert_batchify_for_eval
        return dataloder_dict
            
    raise ValueError('Invalid loss class: %s' % args.loss_class)

from .triple_loader import triple_dataset
from .pair_loader import pair_dataset
from .eval_loader import eval_dataset
from .bert_utils import (
    BertTokenizer, 
    bert_triple_batchify_for_train, 
    bert_pair_batchify_for_train,
    bert_batchify_for_eval
)