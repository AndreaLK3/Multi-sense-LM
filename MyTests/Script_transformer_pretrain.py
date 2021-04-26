import argparse
import os
import Filesystem as F
import torch
from Filesystem import get_standardLM_filename
from Models.TrainingSetup import setup_pretraining_on_WT2
from Models.TextCorpusReader import get_objects
from Models.TrainingAndEvaluation import run_train
import Utils

def test(model_type="transformer"):

    args = argparse.Namespace()
    args.model_type = model_type
    args.use_graph_input = False
    args.learning_rate = 1e-4 # if model_type=="gru" else 1e-5

    args.sp_method = "fasttext"
    args.random_seed = 1

    # ----- Defining the name, and loading the model object -----
    model_name = get_standardLM_filename(args)
    saved_model_fpath = os.path.join(F.FOLDER_SAVEDMODELS, model_name)

    # ----- Load objects: graph, etc.. Currently using default vocabulary sources and sp_method -----
    vocab_sources_ls = [F.WT2, F.SEMCOR]
    sp_method = Utils.SpMethod.FASTTEXT
    gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    objects = get_objects(vocab_sources_ls, sp_method, grapharea_size=32)

    batch_size = 4
    seq_len = 256

    standardLM_model, train_dataloader, valid_dataloader = \
        setup_pretraining_on_WT2(args.model_type, args.use_graph_input,
                        batch_size, seq_len,
                        vocab_sources_ls, sp_method, grapharea_size=32)

    run_train(standardLM_model, train_dataloader, valid_dataloader, learning_rate=args.learning_rate, num_epochs=30, predict_senses=False,
              vocab_sources_ls=vocab_sources_ls, sp_method=sp_method)