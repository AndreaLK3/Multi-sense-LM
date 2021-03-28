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
from Models.TrainingAndEvaluation import evaluation, init_accuracy_dict
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
def run_train(model, train_dataloader, valid_dataloader, learning_rate, num_epochs, predict_senses=True,
              vocab_sources_ls=(F.WT2, F.SEMCOR), sp_method=CE.Method.FASTTEXT):

    # -------------------- Step 1: Setup model --------------------
    Utils.init_logging('Mini-experiment_' + Utils.get_timestamp_month_to_sec() + '.log', loglevel=logging.INFO)
    slc_or_text = train_dataloader.dataset.sensecorpus_or_text

    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate)]

    model.train()
    polysense_thresholds = (2,3,5,10,30)
    polysense_globals_dict= AD.get_polysenseglobals_dict(vocab_sources_ls, sp_method, thresholds=polysense_thresholds)

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    model_forParameters.predict_senses = predict_senses
    hyperparams_str = Loss.write_doc_logging(train_dataloader, model, model_forParameters, learning_rate)
    try:
        logging.info("Using K=" + str(model_forParameters.K))
        logging.info("C=" + str(model_forParameters.num_C))
        logging.info("context_method=" + str(model_forParameters.context_method))
    except Exception:
        pass # no further hyperparameters were specified

    # -------------------- Step 2: Setup flags --------------------
    steps_logging = 500
    overall_step = 0
    starting_time = time()
    best_valid_loss_globals = inf
    best_valid_loss_senses = inf
    best_accuracy_counts = (0,0,0)
    # if with_freezing:
    #    model_forParameters.predict_senses = False

    # torch.autograd.set_detect_anomaly(True) # for debug
    train_dataiter = iter(itertools.cycle(train_dataloader))
    valid_dataiter = iter(itertools.cycle(valid_dataloader))

    # -------------------- Step 3) The training loop, for each epoch --------------------
    try: # to catch KeyboardInterrupt-s and still save the model

        for epoch in range(1,num_epochs+1):

            optimizer = optimizers[-1]  # pick the most recently created optimizer. Useful if using the freezing option

            # -------------------- Step 3a) Initialization --------------------
            sum_epoch_loss_global = 0
            sum_epoch_loss_senses = 0
            sum_epoch_loss_polysenses = 0

            epoch_step = 1
            epoch_senselabeled_tokens = 0
            epoch_polysense_tokens = 0

            predictions_history_dict = init_accuracy_dict(polysense_thresholds)
            verbose = True if (epoch==num_epochs) or (epoch% 200==0) else False # deciding: log prediction output
            flag_earlystop = False

            # -------------------- Step 3b) Evaluation on the validation set --------------------
            if epoch>1:
                logging.info("After training " + str(epoch-1) + " epochs, validation:")
                valid_loss_globals, valid_loss_senses, polysenses_evaluation_loss = \
                    evaluation(valid_dataloader, valid_dataiter, model, slc_or_text, verbose)

                best_valid_loss_globals = min(best_valid_loss_globals, valid_loss_globals)
                best_valid_loss_senses = min(best_valid_loss_senses, valid_loss_senses)

                if epoch == num_epochs:
                    Loss.write_doc_logging(train_dataloader, model, model_forParameters, learning_rate)

                if flag_earlystop:
                    break

            # -------------------- 3d) The training loop in-epoch, over the batches --------------------
            logging.info("\nEpoch n." + str(epoch) + ":")
            for b_idx in range(len(train_dataloader)-1):
                t0 = time()
                batch_input, batch_labels = train_dataiter.__next__()
                batch_input = batch_input.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                # starting operations on one batch
                optimizer.zero_grad()

                # compute loss for the batch
                (losses_tpl, num_sense_instances_tpl) = \
                   Loss.compute_model_loss(model, batch_input, batch_labels, predictions_history_dict,
                                            polysense_globals_dict, slc_or_text, verbose=False)
                loss_global, loss_sense, loss_multisense = losses_tpl
                num_batch_sense_tokens, num_batch_multisense_tokens = num_sense_instances_tpl

                # running sum of the training loss
                sum_epoch_loss_global = sum_epoch_loss_global + loss_global.item()
                if model_forParameters.predict_senses:
                    sum_epoch_loss_senses = sum_epoch_loss_senses + loss_sense.item() * num_batch_sense_tokens
                    epoch_senselabeled_tokens = epoch_senselabeled_tokens + num_batch_sense_tokens
                    sum_epoch_loss_polysenses = sum_epoch_loss_polysenses + loss_multisense.item() * num_batch_multisense_tokens
                    epoch_polysense_tokens = epoch_polysense_tokens + num_batch_multisense_tokens
                    loss = loss_global + loss_sense
                else:
                    loss = loss_global

                loss.backward()

                optimizer.step()
                overall_step = overall_step + 1
                epoch_step = epoch_step + 1

                if overall_step % steps_logging == 0:
                    logging.info("Global step=" + str(overall_step) + "\t ; Iteration time=" + str(round(time()-t0,5)))
                    gc.collect()
                # end of an epoch.

            # -------------------- 3e) Computing training losses for the epoch--------------------
            logging.info("-----\n Models, end of epoch " + str(epoch) + ". Global step n." + str(overall_step) +
                         ". Time = " + str(round(time() - starting_time, 2)))
            logging.info("Models - Correct predictions / Total predictions:\n" + str(predictions_history_dict))

            epoch_sumlosses_tpl = sum_epoch_loss_global, sum_epoch_loss_senses, sum_epoch_loss_polysenses
            epoch_numsteps_tpl = epoch_step, epoch_senselabeled_tokens, epoch_polysense_tokens
            Utils.record_statistics(epoch_sumlosses_tpl, epoch_numsteps_tpl)

    except KeyboardInterrupt:
        logging.info("Models loop interrupted manually by keyboard")