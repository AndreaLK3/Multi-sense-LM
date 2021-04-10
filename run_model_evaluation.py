import argparse
import os
import Filesystem as F
import torch
import logging
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
from Models.TrainingSetup import get_objects, setup_corpus, setup_model
from Models.TrainingAndEvaluation import evaluation
import itertools
import Utils

def parse_arguments():
    parser = argparse.ArgumentParser(description='Creating a model, training it on the sense-labeled corpus.')

    parser.add_argument('--model_type', type=str, choices=['rnn', 'selectk', 'mfs', 'sensecontext', 'selfatt'],
                        help='the model variant used for Multi-sense Language Modeling')

    parser.add_argument('--use_gold_lm', type=bool, default=False,
                        help='Whether the model uses the GoldLM for the Standard LM sub-task: reading ahead the correct next word.')
    parser.add_argument('--predict_senses', type=bool, default=True,
                        help='Whether the model is set up to predict senses and has been trained on it.')
    parser.add_argument('--use_graph_input', type=str, default='no',
                        choices=['no', 'concat', 'replace'],
                        help='Whether to use the GNN input from the dictionary graph alongside the pre-trained word'
                             ' embeddings.')

    parser.add_argument('--sp_method', type=str, default='fasttext', choices=['fasttext', 'transformer'],
                        help="Which method is used to create the single-prototype embeddings: FastText or Transformer")

    # Method-specific parameters
    parser.add_argument('--K', type=int, default=1,
                        help='we choose the correct senses among those of the first top-K predicted words')
    parser.add_argument('--C', type=int, default=20,
                        help='number of previous tokens to average to get the context representation (if used)')
    parser.add_argument('--context_method', type=int, default=0,
                        help='Which context representation to use, in the methods: SenseContext, Self-Attention scores.'
                             ' 0=average of the last C tokens; 1=GRU with 3 layers')

    args = parser.parse_args()
    return args

# The model name was originally created in Models/Loss/write_doc_logging()
def create_modelname_from_args(args):
    model_type = args.model_type.upper()
    model_name = model_type
    if args.use_gold_lm:
        model_name = model_name + "_GoldLM"
    if not(args.predict_senses):
        model_name = model_name + "_noSenses"
    if args.use_graph_input != "no":
        model_name = model_name + "_withGraph"
    if model_type not in ["RNN", "MFS"]:
        model_name = model_name + "_K" + str(args.K)
    if model_type in ["SenseContext".upper(), "SelfAtt".upper()]:
        model_name = model_name + "_C" + str(args.C)
        model_name = model_name + "_ctx" + str(args.context_method)

    return model_name + ".model"


# This module a saved model on SemCor's test split and on the SensEval/SemEval corpus by Raganato et al. 2017
# We specify the parameters of the saved model, that determine its filename

args = parse_arguments()

# ----- Defining the name, and loading the model object -----
model_name = create_modelname_from_args(args)
Utils.init_logging("Evaluation_" + model_name.replace(".model", "") + ".log")

saved_model_fpath = os.path.join(F.FOLDER_MODELS, F.FOLDER_SAVEDMODELS, model_name)
if torch.cuda.is_available():
    model_obj = torch.load(saved_model_fpath)
    if model_obj.__class__.__name__ == "DataParallel":
        model = model_obj.module # unwrap DataParallel
    else:
        model = model_obj
else:
    model = torch.load(saved_model_fpath, map_location=torch.device('cpu')).module  # in case one wishes to use the CPU

logging.info("Loading the model found at: " + str(saved_model_fpath))

# ----- Load objects: graph, etc.. Currently using default vocabulary sources and sp_method -----
vocab_sources_ls = [F.WT2, F.SEMCOR]
sp_method = CE.Method.FASTTEXT
gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
objects = get_objects(vocab_sources_ls, sp_method, grapharea_size=32)

# ----- Setup model: We pass the saved model, so most parameters are unused -----
model, model_forDataLoading, batch_size = setup_model(premade_model=model, model_type=None,
                                                      include_globalnode_input=None, use_gold_lm=None,
                                                      K=None, vocab_sources_ls=vocab_sources_ls, sp_method=sp_method, context_method=None,
                                                      C=None, dim_qkv=None, grapharea_size=32, batch_size=32)

# ----- Setup corpus and evaluate, on SemCor's test split -----
corpus_test_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], F.FOLDER_TEST)
slc_or_text = True
_, test_dataloader = setup_corpus(objects, corpus_test_fpath, slc_or_text, gr_in_voc_folders,
                                               batch_size, seq_len=35, model_forDataLoading=model_forDataLoading)
test_dataiter = iter(itertools.cycle(test_dataloader))
evaluation(test_dataloader, test_dataiter, model, verbose=False)

# ----- Setup corpus and evaluate, on Raganato's SensEval data -----
corpus_test_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SENSEVAL], F.FOLDER_TEST)
slc_or_text = True
_, test_dataloader = setup_corpus(objects, corpus_test_fpath, slc_or_text, gr_in_voc_folders,
                                               batch_size, seq_len=35, model_forDataLoading=model_forDataLoading)
test_dataiter = iter(itertools.cycle(test_dataloader))
evaluation(test_dataloader, test_dataiter, model, verbose=False)