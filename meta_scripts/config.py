import os
import sys
import time
import logging
import argparse

logger = logging.getLogger()


def add_default_args(parser):
    # **************************************
    # Mode
    # **************************************
    modes = parser.add_argument_group("Modes")
    modes.add_argument(
        "--run_mode", 
        choices=["train", "test"],
        type=str,
        help="Training model or testing model.",
    )
    modes.add_argument(
        "--mode_name", 
        choices=["meta", "metafine"],
        type=str,
        help="meta or finetune training/checkpoints.",
    )
    modes.add_argument(
        "--loss_class",
        choices=["pairwise", "pointwise"],
        default="pairwise",
        help="Select learning to rank loss.",
    )
    modes.add_argument(
        "--pretrain_model_type", 
        choices=["bert-base-uncased", "BiomedNLP-PubMedBERT-base-uncased-abstract"],
        default="bert-base-uncased",
        type=str,
        help="Select pretrain model.",
    )
    modes.add_argument(
        "--cv_number",
        type=int, 
        choices=[0, 1, 2, 3, 4],
        help="Select cross validation fold.",
    )
#     modes.add_argument(
#         "--no_cuda", 
#         action="store_true", 
#         default=False,
#         help="Train model on GPUs.",
#     )
    modes.add_argument(
        "--data_workers", 
        default=0, 
        type=int, 
        help="Number of subprocesses for data loading",
    )
    modes.add_argument(
        "--seed", 
        default=42, 
        type=int, 
        help="Set random seed.",
    )
    modes.add_argument(
        "--early_stop_step", 
        default=2000, 
        type=int, 
        help="Early stop steps.",
    )
    # **************************************
    # Files
    # **************************************
    files = parser.add_argument_group("Files")
    files.add_argument("--train_dir", 
                       default=None,
                       type=str, 
                       help="Directory of training data."
                      )
    files.add_argument("--target_dir", 
                       required=True,
                       type=str, 
                       help="Directory of target dataset."
                      )
    files.add_argument("--pretrain_model_dir", 
                       required=True,
                       type=str, 
                       help="Directory of pretrained models."
                      )
    files.add_argument("--save_dir",
                       required=True,
                       type=str, 
                       help="Directory for saving log files, checkpoints, and prediction results."
                      )
    # **************************************
    # Train
    # **************************************
    train = parser.add_argument_group("Train")
    train.add_argument(
        "--do_lower_case", 
        default=True,
        action="store_true", 
        help="Set this flag if you are using an uncased model."
    )
    
    train.add_argument(
        "--max_query_length",
        default=100,
        type=int,
        help="The maximum number of words for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    
    train.add_argument(
        "--max_doc_length",
        default=384,
        type=int,
        help="The maximum total doc sequence length.",
    )
    
    train.add_argument(
        "--max_train_epochs", 
        default=5, 
        type=int,
        help="Total number of training epochs to perform."
    )
    
    train.add_argument("--max_train_steps", 
                       default=-1, 
                       type=int,
                       help="Total number of training steps. ")
    
    train.add_argument("--per_gpu_train_batch_size", 
                       default=8, 
                       type=int,
                       help="Batch size per GPU/CPU for training.")
    train.add_argument(
        "--per_gpu_eval_batch_size", 
        default=256, 
        type=int,
        help="Batch size per GPU/CPU for evaluating."
    )
    
    train.add_argument(
        "--gradient_accumulation_steps", 
        default=8, 
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    
    train.add_argument(
        "--load_checkpoint", 
        default=False,
        action='store_true',
        help="Wether loading checkpoint file."
    )

    train.add_argument(
        "--load_checkpoint_folder", 
        default=None, 
        type=str, 
        help="Load checkpoint model continue training or testing."
    )
    train.add_argument(
        "--load_optimizer", 
        default=False,
        action='store_true',
        help="Whether load optimizer and scheduler files."
    )
    train.add_argument(
        "--checkpoint_name", 
        default="step_best", 
        type=str, 
        help="Load step_n or step_best."
    )
    train.add_argument(
        "--save_checkpoint", 
        default=False, 
        action="store_true", 
        help="Whether save model and optimizer during training."
    )
    
    # **************************************
    # Optimizer
    # **************************************
    optim = parser.add_argument_group("Optimizer")
    optim.add_argument(
        "--learning_rate", 
        default=2e-5, 
        type=float,
        help="The initial learning rate for Adam."
    )
    
    optim.add_argument(
        "--weight_decay", 
        default=0.0, 
        type=float,
        help="Weight deay if we apply some. original = 0.01"
    )
    
    optim.add_argument(
        "--adam_epsilon", 
        default=1e-8, 
        type=float,
        help="Epsilon for Adam optimizer."
    )
    
    optim.add_argument(
        "--num_warmup_steps", 
        default=100, 
        type=int,
        help="Linear warmup steps."
    )
    # **************************************
    # Test
    # **************************************
    test = parser.add_argument_group("Test")    
    test.add_argument(
        "--main_metric", 
        default="ndcg_cut_20", 
        type=str, 
        help="Main metric for testing."
    )
    # **************************************
    # General
    # **************************************
    general = parser.add_argument_group("General")
    general.add_argument(
        "--save_optimizer", 
        default=False, 
        action="store_true", 
        help="Whether save optimizer % scheduler checkpoint"
    )
    general.add_argument(
        "--display_iter", 
        default=16, 
        type=int, 
        help="Log state after every <display_iter> batches."
    )
    
    general.add_argument(
        "--eval_step", 
        default=16, 
        type=int, 
        help="Do evaluation after <eval_steps>."
    )
    general.add_argument(
        "--eval_during_train", 
        default=False, 
        action="store_true",
        help="Run evaluation during training at each eval step."
    )
    general.add_argument(
        "--save_checkpoint_step", 
        default=16, 
        type=int, 
        help="Save checkpoint every <save_checkpoint_step>."
    )
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def init_args_config(args):
    
    # pretrained model dir
    args.cache_pretrain_dir = os.path.join(args.pretrain_model_dir, args.pretrain_model_type)
    
    # target dataset (target clean data or dev)
    target_dataset_name=args.target_dir.split("/")[-1]
    args.target_dir = os.path.join(args.target_dir, "fold_%d"%args.cv_number)
    
    if args.run_mode == "train":
        assert args.train_dir is not None
        # training dataset (weak data or finetune data)
        train_dataset_name = args.train_dir.split("/")[-1] if args.run_mode == "train" else _
     
        # target dataset (clean data)
        if args.mode_name == "meta":
            args.save_name = "%s_%s_%s"%(args.mode_name, train_dataset_name, target_dataset_name)
        else:
            args.train_dir = os.path.join(args.train_dir, "fold_%d"%args.cv_number)
            args.save_name = "%s_%s"%(args.mode_name, train_dataset_name)
    else:
        args.save_name = "%s_%s"%(args.mode_name, target_dataset_name)

    # mkdir results
    if not os.path.exists(args.save_dir):os.mkdir(args.save_dir)
        
    
    # mkdir tot result folder
    args.cv_folder = os.path.join(args.save_dir, "__".join([args.run_mode, args.save_name]))
    if not os.path.exists(args.cv_folder):
        os.mkdir(args.cv_folder)
            
    # save sub-folder for fold_n
    args.save_folder = os.path.join(
        args.cv_folder, 
        "__".join([time.strftime("%m%d-%H%M-%S"), args.save_name, "fold_%d"%args.cv_number])
    )
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
            
    # checkpoint folder
    if args.run_mode == "train":
        args.checkpoint_folder = os.path.join(args.save_folder, "checkpoints")
        if not os.path.exists(args.checkpoint_folder):
            os.mkdir(args.checkpoint_folder)
            
        # viso folder
        args.viso_folder = os.path.join(args.save_folder, "viso")
        if not os.path.exists(args.viso_folder):
            os.mkdir(args.viso_folder)

    # logging file
    args.log_file = os.path.join(args.save_folder, "logging.txt")
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO) # logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
    
    console = logging.StreamHandler() 
    console.setFormatter(fmt) 
    logger.addHandler(console) 
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, "w")
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info("COMMAND: %s" % " ".join(sys.argv))