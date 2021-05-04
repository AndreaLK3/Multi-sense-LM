import argparse
import logging

import Models.TrainingSetup as TS
import Models.TrainingAndEvaluation as TE
from time import time
import Utils
from Filesystem import get_model_name
import copy

def parse_training_arguments():

    parser = argparse.ArgumentParser(description='Creating a model, training it on the sense-labeled corpus.')
    # Necessary parameters
    parser.add_argument('--model_type', type=str, choices=['rnn', 'transformer', 'selectk', 'mfs', 'sensecontext', 'selfatt'],
                        help='model to use for Multi-sense Language Modeling')
    parser.add_argument('--standard_lm', type=str, choices=['gru', 'transformer', 'gold_lm'],
                        help='Which pre-trained instrument to load for standard Language Modeling subtask: '
                             'GRU, Transformer-XL, or reading ahead the correct next word')

    # Optional parameters
    parser.add_argument('--pretrained_senses', type=bool, default=False,
                        help="Whether to load the senses' architecture that was trained with a Gold LM.")
    parser.add_argument('--use_graph_input', type=bool, default=False,
                        help='Whether to use the GNN input from the dictionary graph alongside the pre-trained word'
                             ' embeddings.')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate for training the model (it is a parameter of the Adam optimizer)')

    # Optional parameters that are method-specific
    parser.add_argument('--K', type=int, default=1,
                        help='we choose the correct senses among those of the first top-K predicted words')
    parser.add_argument('--context_method_id', type=int, default=0,
                        help='Which context representation to use, in the methods: SenseContext, Self-Attention scores.'
                             ' 0=average of the last C tokens; 1=GRU with 3 layers')
    parser.add_argument('--C', type=int, default=20,
                        help='number of previous tokens to average to get the context representation (if used)')
    parser.add_argument('--random_seed', type=int, default=1,
                        help='We can specify a random seed != 0 for reproducibility. Default 1')

    args = parser.parse_args()
    return args

args = parse_training_arguments()
Utils.init_logging("Training_" + get_model_name(model=None, args=args).replace(".pt", "") + ".log")

args_to_load_standardlm = copy.deepcopy(args)
args_to_load_standardlm.model_type = "standardlm"
standardLM_model_fname = get_model_name(model=None, args=args_to_load_standardlm)
standardLM_model = TS.load_model_from_file(standardLM_model_fname)
t0 = time()

if args.standard_lm == "transformer":
    batch_size = 2
    seq_len = 256
else:  # GRU and gold_lm
    batch_size = 32
    seq_len = 35

model, train_dataloader, valid_dataloader = TS.setup_training_on_SemCor(standardLM_model, model_type=args.model_type,
                             K=args.K, context_method_id=args.context_method_id, C=args.C,
                             dim_qkv=300, grapharea_size=32, batch_size=batch_size, seq_len=seq_len)

# In case we are loading the senses' architecture that was trained with the Gold_LM StandardLM:
if args.pretrained_senses:
    args_for_goldlm = copy.deepcopy(args)
    args_for_goldlm.standard_lm = "gold_lm"
    try:
        model_with_goldlm = TS.load_model_from_file(get_model_name(model=None, args=args_for_goldlm))
        TS.load_model_senses_architecture(args.model_type, model, model_with_goldlm)
    except FileNotFoundError:
        logging.info("Loading a pre-trained senses' architecture requires pre-training a version of the model " +
                     " that uses the gold_lm as standard language model.")
        raise Exception
    args.learning_rate =  args.learning_rate / 2  # fine-tuning, we halve the learning rate (e.g. 5e-5 -> 2.5e-5)

TE.run_train(model, train_dataloader, valid_dataloader, learning_rate=args.learning_rate, predict_senses=True)

t1 = time() ; Utils.time_measurement_with_msg(t0, t1, "Trained model")