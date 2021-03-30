# This file should mostly copy the Models/TrainingSetup.py and /TrainingAndEvaluation.py files, although
# but instead small mini-experiments (overfitting on a fragment of the training set).
# We operate on separate mini-corpora, and print the input processed by the RNNs forward() call.

import torch
import Utils
import Filesystem as F
import logging
import os
import Graph.Adjacencies as AD
from Models.TrainingSetup import get_objects, setup_corpus, setup_model, ContextMethod, ModelType
from Models.TrainingAndEvaluation import evaluation, init_accuracy_dict, run_train
from Utils import DEVICE
import Models.Loss as Loss
import gc
from math import inf, exp
from time import time
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import Models.ExplorePredictions as EP
import itertools

################ Auxiliary function, for logging ################
def log_input(batch_input, last_idx_senses, vocab_sources_ls, sp_method=CE.Method.FASTTEXT):
    graph_folder, input_folder, vocabulary_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)

    batch_words_ls = []
    batch_senses_ls = []

    # globals:
    batch_globals_indices = batch_input[:, :, 0] - last_idx_senses
    batch_senses_indices = batch_input[:, :, (batch_input.shape[2]//2)]
    for b in range(batch_globals_indices.shape[0]):
        for t in range(batch_globals_indices.shape[1]):
            global_index = batch_globals_indices[b,t].item()
            word = EP.get_globalword_fromindex_df(global_index, vocabulary_folder)
            batch_words_ls.append(word)
    # senses:
    for b in range(batch_globals_indices.shape[0]):
        for t in range(batch_globals_indices.shape[1]):
            sense_index = batch_senses_indices[b,t].item()
            sense = EP.get_sense_fromindex(sense_index, input_folder)
            batch_senses_ls.append(sense)
    logging.info("\n******\nBatch globals: " + str(batch_words_ls))
    logging.info("Batch senses: " + str(batch_senses_ls))
    return


###### Setup model and corpora ######
def setup_training(model_type, include_globalnode_input, use_gold_lm, K,
                load_saved_model=False, sp_method=CE.Method.FASTTEXT, context_method=ContextMethod.AVERAGE, C=0,
                dim_qkv=300, grapharea_size=32, batch_size=4, seq_len=3, vocab_sources_ls=(F.WT2, F.SEMCOR), random_seed=1):
    gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)

    objects = get_objects(vocab_sources_ls, sp_method, grapharea_size)
    # objects == graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix

    model, model_forDataLoading, batch_size = setup_model(model_type, include_globalnode_input, use_gold_lm, K,
                                                                load_saved_model, sp_method, context_method, C,
                                            dim_qkv, grapharea_size, batch_size, vocab_sources_ls, random_seed)

    semcor_train_fpath = os.path.join(F.FOLDER_MINICORPORA, F.FOLDER_SENSELABELED, F.FOLDER_TRAIN)
    semcor_valid_fpath = os.path.join(F.FOLDER_MINICORPORA, F.FOLDER_SENSELABELED, F.FOLDER_VALIDATION)

    train_dataset, train_dataloader = setup_corpus(objects, semcor_train_fpath, True, gr_in_voc_folders,
                                                   batch_size, seq_len, model_forDataLoading)
    valid_dataset, valid_dataloader = setup_corpus(objects, semcor_valid_fpath, True, gr_in_voc_folders,
                                                   batch_size, seq_len, model_forDataLoading)
    return model, train_dataloader, valid_dataloader


##### run loop of training + evaluation ######
# invoke the same run_train() of TrainingAndEvaluation.py