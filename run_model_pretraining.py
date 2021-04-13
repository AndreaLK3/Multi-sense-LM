import argparse
import os
import Filesystem as F
import torch
import logging
from Models.TrainingSetup import get_objects, setup_corpus, setup_pretraining_on_WT2
from Models.TrainingAndEvaluation import run_train
import itertools
import Utils

def parse_arguments():
    parser = argparse.ArgumentParser(description='Creating a StandardLM model, training it on WikiText-2.')

    parser.add_argument('--model_type', type=str, choices=['gru', 'transformer', 'gold_lm'],
                        help='Which instrument to use for standard Language Modeling: GRU, Transformer-XL, '
                             'or reading ahead the correct next word')
    parser.add_argument('--use_graph_input', type=bool, default=False,
                        help='Whether to use the GNN input from the dictionary graph alongside the pre-trained word'
                             ' embeddings.')

    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate for training the model; it is a parameter of the Adam optimizer')
    parser.add_argument('--sp_method', type=str, default='fasttext', choices=['fasttext', 'transformer'],
                        help="Which method is used to create the single-prototype embeddings: FastText or Transformer")
    parser.add_argument('--random_seed', type=int, default=1,
                        help="We can specify a randomization seed !=0, for reproducibility of experiments")


    args = parser.parse_args()
    return args

def get_standardLM_filename(args):
    model_type = "StandardLM_" + args.model_type.lower()
    model_name = model_type
    if args.use_graph_input:
        model_name = model_name + "_withGraph"

    return model_name + ".pt"


args = parse_arguments()

if args.random_seed != 0:
    torch.manual_seed(args.random_seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

# ----- Defining the name, and loading the model object -----
model_name = get_standardLM_filename(args)
Utils.init_logging("Pretraining_" + model_name.replace(".pt", "") + ".log")
saved_model_fpath = os.path.join(F.FOLDER_SAVEDMODELS, model_name)

# ----- Load objects: graph, etc.. Currently using default vocabulary sources and sp_method -----
vocab_sources_ls = [F.WT2, F.SEMCOR]
sp_method = Utils.SpMethod.FASTTEXT
gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
objects = get_objects(vocab_sources_ls, sp_method, grapharea_size=32)

if args.model_type == "transformer":
    batch_size = 1
    seq_len = 512
else: # GRU and gold_lm
    batch_size = 32
    seq_len = 35

standardLM_model, train_dataloader, valid_dataloader = \
    setup_pretraining_on_WT2(args.use_graph_input, args.use_gold_lm, args.use_transformer_lm,
                        batch_size, seq_len,
                        vocab_sources_ls, sp_method, grapharea_size=32)

run_train(standardLM_model, train_dataloader, valid_dataloader, learning_rate=args.learning_rate, num_epochs=30, predict_senses=False,
              vocab_sources_ls=vocab_sources_ls, sp_method=sp_method)