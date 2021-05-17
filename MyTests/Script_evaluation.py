import argparse
import os
import Filesystem as F
import torch
import logging
from Models.TextCorpusReader import get_objects, setup_corpus
from Models.TrainingAndEvaluation import evaluation
import itertools
import Utils

def test():
    args = argparse.Namespace()
    args.model_type = "selectk"
    args.standard_lm = "gold_lm"
    args.use_graph_input = True,
    args.learning_rate = 5e-5
    args.num_epochs = 30
    args.sp_method = "fasttext"
    args.K = 1
    args.context_method_id = 0
    args.C = 20
    args.random_seed = 1

    # ----- Random seed, for reproducibility -----
    if args.random_seed != 0:
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)

    # ----- Defining the name, and loading the model object -----
    model_name = F.get_model_name(model=None, args=args)
    Utils.init_logging("Test-Evaluation_" + model_name.replace(".pt", "") + ".log")

    saved_model_fpath = os.path.join(F.FOLDER_SAVEDMODELS, model_name)
    if torch.cuda.is_available():
        model = torch.load(saved_model_fpath)
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())
        model.to(CURRENT_DEVICE)
    else:
        model = torch.load(saved_model_fpath, map_location=torch.device('cpu'))  # in case one wishes to use the CPU
    logging.info("Loading the model found at: " + str(saved_model_fpath))
    if model.StandardLM.use_transformer_lm:
        batch_size = 2
        seq_len = 256
    else:  # GRU and gold_lm
        batch_size = 16  # 32
        seq_len = 32  # 35

    # ----- Load objects: graph, etc.. Currently using default vocabulary sources and sp_method -----
    vocab_sources_ls = [F.WT2, F.SEMCOR]
    sp_method = Utils.SpMethod.FASTTEXT
    gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    objects = get_objects(vocab_sources_ls, sp_method, grapharea_size=32)

    # ----- Setup corpus and evaluate, on Raganato's SensEval data -----
    corpus_test_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SENSEVAL], F.FOLDER_TEST)
    slc_or_text = True
    _, test_dataloader = setup_corpus(objects, corpus_test_fpath, slc_or_text, gr_in_voc_folders, batch_size, seq_len)
    test_dataiter = iter(itertools.cycle(test_dataloader))
    evaluation(test_dataloader, test_dataiter, model, verbose=True)