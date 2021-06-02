from ..transformers import AutoConfig

def get_class(args):
    config = AutoConfig.from_pretrained(
        args.cache_pretrain_dir, 
        num_labels=args.num_labels)

    return BertRanker.from_pretrained(
        args.cache_pretrain_dir, 
        config=config,
        loss_class=args.loss_class,)

from .bert_ranker import BertRanker
from .magic_module import MagicModule