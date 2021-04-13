import argparse
import os
import Filesystem as F
import torch
import logging
import run_model_training as rmt
from Models.TrainingSetup import get_objects, setup_corpus
from Models.TrainingAndEvaluation import evaluation
import itertools
import Utils

# This module a saved model on SemCor's test split and on the SensEval/SemEval corpus by Raganato et al. 2017
# We specify the parameters of the saved model, that determine its filename

args = rmt.parse_arguments()

# ----- Random seed, for reproducibility -----
if args.random_seed != 0:
    torch.manual_seed(args.random_seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)


# ----- Defining the name, and loading the model object -----
model_name = F.get_model_name(model=None, args=args)
Utils.init_logging("Evaluation_" + model_name.replace(".pt", "") + ".log")

saved_model_fpath = os.path.join(F.FOLDER_SAVEDMODELS, model_name)
if torch.cuda.is_available():
    model = torch.load(saved_model_fpath)
else:
    model = torch.load(saved_model_fpath, map_location=torch.device('cpu')).module  # in case one wishes to use the CPU
logging.info("Loading the model found at: " + str(saved_model_fpath))
if model.use_transformer_lm:
    batch_size = 1
    seq_len = 512
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