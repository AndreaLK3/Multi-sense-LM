# This file should mostly copy the NN/Training.py file, although it is not used to set up standard experiments,
# but instead small mini-experiments (overfitting on a fragment of the training set).
# We operate on separate mini-corpuses, and print the input processed by the RNNs forward() call.

import torch
import Utils
import Filesystem as F
import logging
import Graph.DefineGraph as DG
import sqlite3
import os
import pandas as pd
from math import inf
import Graph.Adjacencies as AD
import numpy as np
from NN.Training import load_model_from_file
from VocabularyAndEmbeddings.ComputeEmbeddings import Method
from NN.Loss import write_doc_logging, compute_model_loss
from Utils import DEVICE
import NN.DataLoading as DL
import NN.Models.RNNs as RNNs
from itertools import cycle
import gc
from math import exp
from time import time
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import NN.ExplorePredictions as EP


################ Auxiliary function, for logging ################
def log_input(batch_input, last_idx_senses, slc_or_text):
    subfolder = F.FOLDER_SENSELABELED if slc_or_text else F.FOLDER_STANDARDTEXT
    inputdata_folder = os.path.join(F.FOLDER_INPUT, subfolder)
    vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, subfolder)

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
    if slc_or_text:
        for b in range(batch_globals_indices.shape[0]):
            for t in range(batch_globals_indices.shape[1]):
                sense_index = batch_senses_indices[b,t].item()
                sense = EP.get_sense_fromindex(sense_index, inputdata_folder)
                batch_senses_ls.append(sense)
    logging.info("\n******\nBatch globals: " + str(batch_words_ls))
    logging.info("Batch senses: " + str(batch_senses_ls))
    return



################ Creating the model, the train_dataloader, and any necessary variables ################
def setup_train(slc_or_text_corpus, include_globalnode_input, load_saved_model, grapharea_size, batch_size, sequence_length,
                method=CE.Method.FASTTEXT, allow_dataparallel=True):

    # -------------------- Setting up the graph, grapharea_matrix and vocabulary --------------------
    graph_dataobj = DG.get_graph_dataobject(new=False, method=method, slc_corpus=slc_or_text_corpus).to(DEVICE)

    subfolder = F.FOLDER_SENSELABELED if slc_or_text_corpus else F.FOLDER_STANDARDTEXT
    graph_folder = os.path.join(F.FOLDER_GRAPH, subfolder)
    inputdata_folder = os.path.join(F.FOLDER_INPUT, subfolder)
    vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, subfolder)
    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, grapharea_size, hops_in_area=1, graph_folder=graph_folder)

    single_prototypes_file = F.SPVs_FASTTEXT_FILE if method == Method.FASTTEXT else F.SPVs_DISTILBERT_FILE
    embeddings_matrix = torch.tensor(np.load(os.path.join(inputdata_folder, single_prototypes_file))).to(torch.float32)

    globals_vocabulary_fpath = os.path.join(vocabulary_folder, F.VOCABULARY_OF_GLOBALS_FILENAME)
    vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')

    # -------------------- Loading / creating the model --------------------
    torch.manual_seed(1) # for reproducibility while conducting mini-experiments
    if torch.cuda.is_available():
         torch.cuda.manual_seed_all(1)
    if load_saved_model:
        model = load_model_from_file(slc_or_text_corpus, inputdata_folder, graph_dataobj)
    else:
        model = RNNs.RNN(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                           embeddings_matrix, include_globalnode_input,
                           batch_size=batch_size, n_layers=3, n_hid_units=1024)
        # model = Senses.SelectK(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
        #          include_globalnode_input, batch_size=batch_size, n_layers=3, n_hid_units=1024, k=10)
        # model = Senses.ContextSim(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
        #                            batch_size, seq_len=sequence_length, n_layers=3, n_hid_units=1024, k=10, c=10)

    # -------------------- Moving objects on GPU --------------------
    logging.info("Graph-data object loaded, model initialized. Moving them to GPU device(s) if present.")

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1 and allow_dataparallel:
        logging.info("Using " + str(n_gpu) + " GPUs")
        model = torch.nn.DataParallel(model, dim=0)
        model_forDataLoading = model.module
        batch_size = n_gpu if batch_size is None else batch_size  # if not specified, default batch_size = n. GPUs
    else:
        model_forDataLoading = model
        batch_size = 1 if batch_size is None else batch_size
    model.to(DEVICE)

    # -------------------- Creating the DataLoaders for training and validation dataset --------------------
    senseindices_db_filepath = os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()

    bptt_collator = DL.BPTTBatchCollator(grapharea_size, sequence_length)

    vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')
    train_corpus_fpath = os.path.join(F.FOLDER_MINICORPUSES, subfolder, F.FOLDER_TRAIN)
    train_dataset = DL.TextDataset(slc_or_text_corpus, train_corpus_fpath, senseindices_db_c, vocab_h5, model_forDataLoading,
                                   grapharea_matrix, grapharea_size, graph_dataobj)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size * sequence_length,
                                                   num_workers=0, collate_fn=bptt_collator)

    return model, train_dataloader


################
def run_train(model,train_dataloader, learning_rate, num_epochs, predict_senses, with_freezing):

    # -------------------- Setup; parameters and utilities --------------------
    Utils.init_logging('MiniExp-' + Utils.get_timestamp_month_to_min() + '.log', loglevel=logging.INFO)
    slc_or_text = train_dataloader.dataset.sensecorpus_or_text

    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate)]

    model.train()
    training_losses_lts = []
    multisense_globals_set = set(AD.get_multisense_globals_indices(slc_or_text))

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    model_forParameters.predict_senses = predict_senses

    steps_logging = 50
    overall_step = 0
    starting_time = time()

    freezing_epoch = (num_epochs // 3)*2 # we decide to freeze at 2/3rds of the number of epochs in the miniexperiment
    after_freezing_flag = False
    if with_freezing:
        model_forParameters.predict_senses = False
    weights_before_freezing_check_ls = []
    parameters_to_check_names_ls = []


    # debug
    # torch.autograd.set_detect_anomaly(True)

    train_dataiter = iter(cycle(train_dataloader))


    # -------------------- The training loop --------------------
    try: # to catch KeyboardInterrupt-s and still save the training & validation losses

        for epoch in range(1,num_epochs+1):

            optimizer = optimizers[-1] # pick the most recently created optimizer. Useful when freezing

            # -------------------- Initialization --------------------
            sum_epoch_loss_global = 0
            sum_epoch_loss_sense = 0
            sum_epoch_loss_multisense = 0

            epoch_step = 0
            epoch_senselabeled_tokens = 0
            epoch_multisense_tokens = 0

            correct_predictions_dict = {'correct_g':0,
                                        'top_k_g':0,
                                        'tot_g':0,
                                        'correct_all_s':0,
                                        'top_k_all_s':0,
                                        'tot_all_s':0,
                                        'correct_multi_s':0,
                                        'top_k_multi_s':0,
                                        'tot_multi_s':0
                                        }
            verbose = True if (epoch==num_epochs) or (epoch% 200==0) else False # - log prediction output
            # verbose_valid = True if (epoch == num_epochs) or (epoch in []) else False  # - log prediction output


            # -------------------- The training loop over the batches --------------------
            logging.info("\nEpoch n." + str(epoch) + ":")
            for b_idx in range(len(train_dataloader)-1):
                t0 = time()
                batch_input, batch_labels = train_dataiter.__next__()
                batch_input = batch_input.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                if epoch == 1 or epoch==2:
                    log_input(batch_input, model_forParameters.last_idx_senses, slc_or_text)
                # starting operations on one batch
                optimizer.zero_grad()

                # compute loss for the batch
                (losses_tpl, num_sense_instances_tpl) = compute_model_loss(model, batch_input, batch_labels, correct_predictions_dict,
                                                                           multisense_globals_set,slc_or_text, verbose)
                loss_global, loss_sense, loss_multisense = losses_tpl
                num_batch_sense_tokens, num_batch_multisense_tokens = num_sense_instances_tpl

                # running sum of the training loss
                sum_epoch_loss_global = sum_epoch_loss_global + loss_global.item()
                if model_forParameters.predict_senses:
                    sum_epoch_loss_sense = sum_epoch_loss_sense + loss_sense.item() * num_batch_sense_tokens
                    epoch_senselabeled_tokens = epoch_senselabeled_tokens + num_batch_sense_tokens
                    sum_epoch_loss_multisense = sum_epoch_loss_multisense + loss_multisense.item() * num_batch_multisense_tokens
                    epoch_multisense_tokens = epoch_multisense_tokens + num_batch_multisense_tokens
                    if not after_freezing_flag:
                        loss = loss_global + loss_sense
                    else:
                        loss = loss_sense
                else:
                    loss = loss_global

                loss.backward()

                optimizer.step()
                overall_step = overall_step + 1
                epoch_step = epoch_step + 1

                if overall_step % steps_logging == 0:
                    logging.info("Global step=" + str(overall_step) + "\t ; Iteration time=" + str(round(time()-t0,5)))
                    gc.collect()

                if epoch == freezing_epoch and not after_freezing_flag:
                    # we are predicting first the standard LM, and then the senses. Freeze (1), activate (2).
                    logging.info("New validation worse than previous one. " +
                                 "Freezing the weights in the standard LM, activating senses' prediction.")
                    for (name, p) in model_forParameters.named_parameters():  # (1)
                        parameters_to_check_names_ls.append(name)
                        weights_before_freezing_check_ls.append(p.clone().detach())

                        if ("main_rnn" in name) or ("E" in name) or ("X" in name) or ("linear2global" in name):
                            logging.info(name)
                            p.requires_grad = False
                            p.grad = p.grad * 0
                    optimizers.append(torch.optim.Adam(model.parameters(),
                                                       lr=learning_rate))  # [p for p in model.parameters() if p.requires_grad]
                    optimizer = optimizers[-1]  # pick the most recently created optimizer
                    model_forParameters.predict_senses = True  # (2)
                    after_freezing_flag = True

                # end of an epoch.

            # -------------------- Computing training losses for the epoch--------------------
            logging.info("-----\n Training, end of epoch " + str(epoch) + ". Global step n." + str(overall_step) +
                         ". Time = " + str(round(time() - starting_time, 2)))
            logging.info("Training - Correct predictions / Total predictions:\n" + str(correct_predictions_dict))

            epoch_sumlosses_tpl = sum_epoch_loss_global, sum_epoch_loss_sense, sum_epoch_loss_multisense
            epoch_numsteps_tpl = epoch_step, epoch_senselabeled_tokens, epoch_multisense_tokens
            Utils.record_statistics(epoch_sumlosses_tpl, epoch_numsteps_tpl, training_losses_lts)


    except KeyboardInterrupt:
        logging.info("Training loop interrupted manually by keyboard")