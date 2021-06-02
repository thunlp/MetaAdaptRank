import os
import sys
import time
import logging
import argparse

logger = logging.getLogger()

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def add_default_args(parser):
    
    ## ************************
    # Modes
    ## ************************
    modes = parser.add_argument_group("Modes")
    modes.add_argument(
        "--run_mode", 
        choices=["train", "inference"],
        type=str,
        help="Training nlg model or inference.",
    )
    modes.add_argument(
        "--generator_mode", 
        choices=["qg", "contrastqg"],
        required=True,
        type=str, 
        help="Select contrastqg or qg mode",
    )
    
    modes.add_argument(
        "--pretrain_generator_type", 
        choices=["t5-small", "t5-base"],
        default="t5-small",
        type=str,
        help="Select pretrain generator type.",
    )
    modes.add_argument(
        "--no_cuda", 
        action="store_true", 
        default=False,
        help="Train model on GPUs.",
    )
    modes.add_argument(
        "--local_rank", 
        default=-1, 
        type=int, 
        help="Set local_rank=0 for distributed training on multiple gpus.",
    )
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
        help="Random seed for initialization: 42",
    )
    
    ## ************************
    # Train
    ## ************************
    train = parser.add_argument_group("Train")

    train.add_argument("--max_train_steps", 
                       default=5000000, 
                       type=int,
                       help="Total number of training steps.")
    train.add_argument(
        "--per_gpu_train_batch_size", 
        default=4, 
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    train.add_argument(
        "--gradient_accumulation_steps", 
        default=1, 
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass."
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
    
    ## ************************
    # File
    ## ************************
    files = parser.add_argument_group("Files")
    files.add_argument(
        "--train_file", 
        default=None,
        type=str, 
        help="Training file path.",
    )
    files.add_argument(
        "--target_dataset_dir", 
        default=None,
        type=str, 
        help="Target dataset path",
    )
    files.add_argument(
        "--pretrain_model_dir", 
        type=str, 
        default=None,
        help="Directory of pretrained models."
    )
    files.add_argument(
        "--save_dir",
        required=True,
        type=str, 
        help="Directory for saving log files, checkpoints, and prediction results."
    )
    
    ## ************************
    # Inference
    ## ************************
    inference = parser.add_argument_group("Inference")
    inference.add_argument(
        "--per_gpu_gen_batch_size", 
        default=64, 
        type=int,
        help="Batch size per GPU/CPU for test."
    )
    inference.add_argument(
        "--generator_load_dir", 
        type=str, 
        default=None,
    )
    inference.add_argument(
        "--reverse_genseq", 
        action='store_true', 
        default=False
    )
    
    # **************************************
    # General
    # **************************************
    general = parser.add_argument_group("General")
    general.add_argument(
        "--max_input_length", 
        type=int, 
        default=512
    )
    general.add_argument(
        "--max_gen_len", 
        type=int, 
        default=32, 
        help="Maximum length of output sequence"
    )
    general.add_argument(
        "--min_gen_length", 
        type=int, 
        default=20
    )
    general.add_argument(
        "--temperature", 
        type=float, 
        default=1.0, 
        help="temperature of 1 implies greedy sampling. \
        The value used to module the next token probabilities. Must be strictly positive"
    )
    general.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="The cumulative probability of parameter highest probability \
        vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1."
    )
    general.add_argument(
        "--retry_times", 
        type=int, 
        default=3
    )
    general.add_argument(
        "--display_iter", 
        default=16, 
        type=int, 
        help="Log state after every <display_iter> batches."
    )
    general.add_argument(
        "--save_checkpoint_step", 
        default=50000, 
        type=int, 
        help="Save checkpoint every <save_checkpoint_step>."
    )
    
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def init_args_config(args):
    
    if args.run_mode == "train":
        # pretrained model dir
        args.generator_load_dir = os.path.join(args.pretrain_model_dir, args.pretrain_generator_type)
    
    # mkdir results
    if not os.path.exists(args.save_dir):os.mkdir(args.save_dir)
        
    args.save_folder = os.path.join(args.save_dir,  "__".join([args.run_mode, args.generator_mode]))
    if not os.path.exists(args.save_folder):os.mkdir(args.save_folder)
        
    args.save_folder = os.path.join(args.save_folder, "__".join([args.pretrain_generator_type, time.strftime("%m%d-%H%M-%S")]))
    if not os.path.exists(args.save_folder):os.mkdir(args.save_folder)
        
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