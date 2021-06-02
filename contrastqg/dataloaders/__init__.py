def select_tokenizer(args): 
    if "t5" in args.pretrain_generator_type:
        return T5_Tokenizer(args)
    raise ValueError('Invalid generator class: %s' % args.pretrain_generator_type)
    
    
def select_data_loader(args):
    if "train" in args.run_mode:
        dataloder_dict = {"train_dataset":train_generate_dataset, "train_batchify":t5_batchify_for_train}
        return dataloder_dict
    else:
        dataloder_dict = {"build_generate_dataset":generate_dataset}
        if "t5" in args.pretrain_generator_type:
            dataloder_dict["gen_batchify"] = t5_batchify_for_test
            return dataloder_dict
        raise ValueError('Invalid generator class: %s' % args.pretrain_generator_type)
    raise ValueError('Invalid run mode: [%s]' % args.run_mode)
    
from .train_generate_loader import train_generate_dataset

    
# def select_data_loader(args):
#     if "train" in args.run_mode:
#         dataloder_dict = {"train_dataset":train_generate_dataset, "train_loader":query_generator_train_dataloader}
#         return dataloder_dict
#     else:
#         dataloder_dict = {"build_generate_dataset":generate_dataset}
#         if "t5" in args.pretrain_generator_type:
#             dataloder_dict["gen_batchify"] = t5_batchify_for_test
#             return dataloder_dict
#         raise ValueError('Invalid generator class: %s' % args.pretrain_generator_type)
#     raise ValueError('Invalid run mode: [%s]' % args.run_mode)
# from .train_generate_loader import train_generate_dataset, query_generator_train_dataloader


from .generate_loader import generate_dataset

from .t5_utils import (
    T5_Tokenizer,
    t5_batchify_for_test,
    t5_batchify_for_train,
)
