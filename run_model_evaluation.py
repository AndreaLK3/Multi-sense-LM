import argparse
import os
import Filesystem as F
import torch
import logging
from Models.TextCorpusReader import get_objects, setup_corpus
from Models.TrainingAndEvaluation import evaluation
import itertools
import Utils

# This module a saved model on SemCor's test split and on the SensEval/SemEval corpus by Raganato et al. 2017
# We specify the parameters of the saved model, that determine its filename

# Identical to the version in run_model_training.py
def parse_training_arguments():

    parser = argparse.ArgumentParser(description='Creating a model, training it on the sense-labeled corpus.')
    # Necessary parameters
    parser.add_argument('--model_type', type=str, choices=['rnn', 'selectk', 'mfs', 'sensecontext', 'selfatt'],
                        help='model to use for Multi-sense Language Modeling')
    parser.add_argument('--standard_lm', type=str, choices=['gru', 'transformer', 'gold_lm'],
                        help='Which pre-trained instrument to load for standard Language Modeling subtask: '
                             'GRU, Transformer-XL, or reading ahead the correct next word')

    # Optional parameters
    parser.add_argument('--use_graph_input', type=bool, default=False,
                        help='Whether to use the GNN input from the dictionary graph alongside the pre-trained word'
                             ' embeddings.')
    parser.add_argument('--learning_rate', type=float, default=0.00005,
                        help='learning rate for training the model (it is a parameter of the Adam optimizer)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='maximum number of epochs for model training. It generally stops earlier because it uses '
                             'early-stopping on the validation set')
    parser.add_argument('--sp_method', type=str, default='fasttext', choices=['fasttext', 'transformer'],
                        help="Which method is used to create the single-prototype embeddings: FastText or Transformer")

    # Optional parameters that are method-specific
    parser.add_argument('--K', type=int, default=1,
                        help='we choose the correct senses among those of the first top-K predicted words')
    parser.add_argument('--context_method_id', type=int, default=0,
                        help='Which context representation to use, in the methods: SenseContext, Self-Attention scores.'
                             ' 0=average of the last C tokens; 1=GRU with 3 layers')
    parser.add_argument('--C', type=int, default=20,
                        help='number of previous tokens to average to get the context representation (if used)')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='We can specify a random seed != 0 for reproducibility')

    args = parser.parse_args()
    return args


args = parse_training_arguments()
Utils.init_logging("starting_run_model_evaluation.log")

# ----- Random seed, for reproducibility -----
if args.random_seed != 0:
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

# ----- Defining the name, and loading the model object -----
model_name = F.get_model_name(model=None, args=args)
Utils.init_logging("Evaluation_" + model_name.replace(".pt", "") + ".log")

saved_model_fpath = os.path.join(F.FOLDER_SAVEDMODELS, model_name)
if torch.cuda.is_available():
    model = torch.load(saved_model_fpath)
    CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())
    model.to(CURRENT_DEVICE)
else:
    model = torch.load(saved_model_fpath, map_location=torch.device('cpu'))  # in case one wishes to use the CPU
logging.info("Loading the model found at: " + str(saved_model_fpath))
if model.StandardLM.use_transformer_lm:
    batch_size = 4
    seq_len = 256
else: # GRU and gold_lm
    batch_size = 32
    seq_len = 35

# ----- Load objects: graph, etc.. Currently using default vocabulary sources and sp_method -----
vocab_sources_ls = [F.WT2, F.SEMCOR]
sp_method = Utils.SpMethod.FASTTEXT
gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
objects = get_objects(vocab_sources_ls, sp_method, grapharea_size=32)


# ----- Setup corpus and evaluate, on SemCor's test split -----
corpus_test_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], F.FOLDER_TEST)
slc_or_text = True
_, test_dataloader =  setup_corpus(objects, corpus_test_fpath, slc_or_text, gr_in_voc_folders, batch_size, seq_len)
test_dataiter = iter(itertools.cycle(test_dataloader))
evaluation(test_dataloader, test_dataiter, model, verbose=False)

# ----- Setup corpus and evaluate, on Raganato's SensEval data -----
corpus_test_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SENSEVAL], F.FOLDER_TEST)
slc_or_text = True
_, test_dataloader =  setup_corpus(objects, corpus_test_fpath, slc_or_text, gr_in_voc_folders, batch_size, seq_len)
test_dataiter = iter(itertools.cycle(test_dataloader))
evaluation(test_dataloader, test_dataiter, model, verbose=False)