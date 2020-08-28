import torch
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tfunc
import Graph.DefineGraph as DG
import sqlite3
import os
import pandas as pd
from math import inf
import Graph.Adjacencies as AD
import numpy as np
from time import time
from Utils import DEVICE
import GNN.DataLoading as DL
import GNN.ExplorePredictions as EP
import GNN.Models.PreviousRNN as PrevRNN
import GNN.Models.RNNs as RNNFreezer
from itertools import cycle
import gc
from math import exp

# This code should be *before* we import torch in order to work and limit correctly which devices we wish to use
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# Preliminary logging, to document the hyperparameters, the model, and its parameters #
def write_doc_logging(train_dataloader, model, model_forParameters, learning_rate, num_epochs):
    hyperparams_str = '_batchPerSeqlen' + str(train_dataloader.batch_size) \
                      + '_area' + str(model_forParameters.grapharea_size)\
                      + '_lr' + str(learning_rate) \
                      + '_epochs' + str(num_epochs)
    logging.info("Hyperparameters: " + hyperparams_str)
    logging.info("Model:")
    logging.info(str(model))
    logging.info("Parameters:")
    parameters_list = [(name, param.shape, param.dtype, param.requires_grad) for (name, param) in model.named_parameters()]
    logging.info('\n'.join([str(p) for p in parameters_list]))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info("Number of trainable parameters=" + str(params))
    return hyperparams_str

##########




def update_predictions_history_dict(correct_preds_dict, predictions_globals, predictions_senses, batch_labels_tpl):

    k = 10
    batch_labels_globals = batch_labels_tpl[0]
    batch_labels_all_senses = batch_labels_tpl[1]
    batch_labels_multi_senses = batch_labels_tpl[2]

    (values_g, indices_g) = predictions_globals.sort(dim=1, descending=True)

    correct_preds_dict['correct_g'] = \
        correct_preds_dict['correct_g'] + torch.sum(indices_g[:, 0] == batch_labels_globals).item()
    correct_preds_dict['tot_g'] = correct_preds_dict['tot_g'] + batch_labels_globals.shape[0]

    top_k_predictions_g = indices_g[:, 0:k]
    batch_counter_top_k_g = 0
    for i in range(len(batch_labels_globals)):
        label_g = batch_labels_globals[i]
        if label_g in top_k_predictions_g[i]:
            batch_counter_top_k_g = batch_counter_top_k_g+1
    correct_preds_dict['top_k_g'] = correct_preds_dict['top_k_g'] + batch_counter_top_k_g


    if len(predictions_senses.shape) > 1:
        (values_s, indices_s) = predictions_senses.sort(dim=1, descending=True)
        correct_preds_dict['correct_all_s'] = \
            correct_preds_dict['correct_all_s'] + torch.sum(indices_s[:, 0] == batch_labels_all_senses).item()
        correct_preds_dict['correct_multi_s'] = \
            correct_preds_dict['correct_multi_s'] + torch.sum(indices_s[:, 0] == batch_labels_multi_senses).item()
        correct_preds_dict['tot_all_s'] = correct_preds_dict['tot_all_s'] + \
                                      (batch_labels_all_senses[batch_labels_all_senses != -1].shape[0])
        correct_preds_dict['tot_multi_s'] = correct_preds_dict['tot_multi_s'] + \
                                          (batch_labels_all_senses[batch_labels_multi_senses != -1].shape[0])

        top_k_predictions_s = indices_s[:, 0:k]
        batch_counter_top_k_all_s = 0
        batch_counter_top_k_multi_s = 0
        for i in range(len(batch_labels_all_senses)):
            label_all_s = batch_labels_all_senses[i]
            label_multi_s = batch_labels_multi_senses[i]
            if label_all_s in top_k_predictions_s[i]:
                batch_counter_top_k_all_s = batch_counter_top_k_all_s + 1
            if label_multi_s in top_k_predictions_s[i]:
                batch_counter_top_k_multi_s = batch_counter_top_k_multi_s + 1
        correct_preds_dict['top_k_all_s'] = correct_preds_dict['top_k_all_s'] + batch_counter_top_k_all_s
        correct_preds_dict['top_k_multi_s'] = correct_preds_dict['top_k_multi_s'] + batch_counter_top_k_multi_s

    logging.debug("updated_predictions_history_dict = " +str(correct_preds_dict))
    return


def compute_model_loss(model, batch_input, batch_labels, correct_preds_dict, multisense_globals_set, verbose=False):

    predictions_globals, predictions_senses = model(batch_input)

    batch_labels_t = (batch_labels).clone().t().to(DEVICE)
    batch_labels_globals = batch_labels_t[0]
    batch_labels_all_senses = batch_labels_t[1]
    batch_labels_multi_senses_ls = list(map(
        lambda i : batch_labels_all_senses[i] if batch_labels_globals[i].item() in multisense_globals_set
                                              else -1, range(len(batch_labels_all_senses))))
    batch_labels_multi_senses = torch.tensor(batch_labels_multi_senses_ls).to(DEVICE)

    # compute the loss for the batch
    loss_global = tfunc.nll_loss(predictions_globals, batch_labels_globals)

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    if model_forParameters.predict_senses:
        loss_all_senses = tfunc.nll_loss(predictions_senses, batch_labels_all_senses, ignore_index=-1)
        loss_multi_senses = tfunc.nll_loss(predictions_senses, batch_labels_multi_senses, ignore_index=-1)
    else:
        loss_all_senses = torch.tensor(0)
        loss_multi_senses = torch.tensor(0)
    # Added to measure the senses' task, given that we can not rely on the senses' PPL for SelectK
    batch_labels_tpl = (batch_labels_globals, batch_labels_all_senses, batch_labels_multi_senses)
    update_predictions_history_dict(correct_preds_dict, predictions_globals, predictions_senses, batch_labels_tpl)

    # debug: check the solutions and predictions. Is there anything the model is unable to predict?
    if verbose:
        logging.info("*******\ncompute_model_loss > verbose logging of batch")
        EP.log_batch(batch_labels, predictions_globals, predictions_senses, 2)

    losses_tpl = loss_global, loss_all_senses, loss_multi_senses
    senses_in_batch = len(batch_labels_all_senses[batch_labels_all_senses != -1])
    multisenses_in_batch = len(batch_labels_multi_senses[batch_labels_multi_senses != -1])
    num_sense_instances_tpl = senses_in_batch, multisenses_in_batch

    return (losses_tpl, num_sense_instances_tpl)



################

def training_setup(slc_or_text_corpus, include_globalnode_input, include_sensenode_input, predict_senses,
                   method, grapharea_size, batch_size, sequence_length, allow_dataparallel=True):
    graph_dataobj = DG.get_graph_dataobject(new=False, method=method, slc_corpus=slc_or_text_corpus).to(DEVICE)
    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, grapharea_size, hops_in_area=1)

    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE)
    globals_vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')
    vocabulary_wordList = globals_vocabulary_df['word'].to_list().copy()
    if slc_or_text_corpus:
        vocabulary_numSensesList = globals_vocabulary_df['num_senses'].to_list().copy()
        if all([num_senses == -1 for num_senses in vocabulary_numSensesList]):
            globals_vocabulary_df = AD.compute_globals_numsenses(graph_dataobj, grapharea_matrix, grapharea_size)

    # torch.manual_seed(1) # for reproducibility while conducting mini-experiments
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(1)
    model = RNNFreezer.RNN("GRU", graph_dataobj, grapharea_size, grapharea_matrix, globals_vocabulary_df,
                        include_globalnode_input, include_sensenode_input, predict_senses,
                        batch_size=batch_size, n_layers=3, n_hid_units=1024, dropout_p=0)
    # model = PrevRNN.RNN(model_type="GRU", data=graph_dataobj, grapharea_size=grapharea_size, grapharea_matrix=grapharea_matrix,
    #                     vocabulary_wordlist=slc_or_text_corpus, include_globalnode_input=include_globalnode_input,
    #                     include_sensenode_input=include_sensenode_input, predict_senses=predict_senses,
    #              batch_size=batch_size, n_layers=3, n_hid_units=1024, dropout_p=0)

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

    senseindices_db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()

    bptt_collator = DL.BPTTBatchCollator(grapharea_size, sequence_length)

    vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')
    train_dataset = DL.TextDataset(slc_or_text_corpus, Utils.TRAINING, senseindices_db_c, vocab_h5, model_forDataLoading,
                                   grapharea_matrix, grapharea_size, graph_dataobj)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size * sequence_length,
                                                   num_workers=0, collate_fn=bptt_collator)

    valid_dataset = DL.TextDataset(slc_or_text_corpus, Utils.VALIDATION, senseindices_db_c, vocab_h5, model_forDataLoading,
                                   grapharea_matrix, grapharea_size, graph_dataobj)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size * sequence_length,
                                                   num_workers=0, collate_fn=bptt_collator)

    return model, train_dataloader, valid_dataloader


################
def training_loop(model, learning_rate, train_dataloader, valid_dataloader, num_epochs, with_freezing):

    Utils.init_logging('Training' + Utils.get_timestamp_month_to_min() + '.log', loglevel=logging.INFO)

    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate)]

    model.train()
    training_losses_lts = [] # mutated into a lts, with (global_loss, sense_loss)
    validation_losses_lts = []
    multisense_globals_set = set(AD.get_multisense_globals_indices())

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    hyperparams_str = write_doc_logging(train_dataloader, model, model_forParameters, learning_rate, num_epochs)

    steps_logging = 100
    overall_step = 0
    starting_time = time()
    best_valid_loss_globals = inf
    after_freezing_flag = False
    if with_freezing:
        model_forParameters.predict_senses = False

    try: # to catch KeyboardInterrupt-s and still save the training & validation losses
        ########## The training loop ##########
        train_dataiter = iter(cycle(train_dataloader))
        valid_dataiter = iter(cycle(valid_dataloader))
        for epoch in range(1,num_epochs+1):
            optimizer = optimizers[-1]
            logging.info("\nTraining epoch n."+str(epoch) + ":")
            if epoch == 300:
                logging.info("debug of correct predictions")

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
            verbose = True if (epoch==num_epochs) or (epoch% 100==0) else False # - log prediction output
            #verbose_valid = True if (epoch == num_epochs) or (epoch % 10 == 0) else False  # - log prediction output

            flag_earlystop = False

            for b_idx in range(len(train_dataloader)-1):
                t0 = time()
                batch_input, batch_labels = train_dataiter.__next__()
                batch_input = batch_input.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                # starting operations on one batch
                optimizer.zero_grad()

                # compute loss for the batch
                (losses_tpl, num_sense_instances_tpl) = compute_model_loss(model, batch_input, batch_labels, correct_predictions_dict,
                                                             multisense_globals_set, verbose)
                loss_global, loss_sense, loss_multisense = losses_tpl
                num_batch_sense_tokens, num_batch_multisense_tokens = num_sense_instances_tpl

                # running sum of the training loss in the log segment
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

                #logging.info("Iteration in training_loop(), time analysis:")
                #Utils.log_chronometer([t0,t1,t2,t3,t4])
                if overall_step % steps_logging == 0:
                    logging.info("Global step=" + str(overall_step) + "\t ; Iteration time=" + str(round(time()-t0,5)))
                    gc.collect()
            # except StopIteration: the DataLoader naturally catches StopIteration
                # end of an epoch.

            logging.info("-----\n Training, end of epoch " + str(epoch) + ". Global step n." + str(overall_step) +
                         ". Time = " + str(round(time() - starting_time, 2)) + ". The training losses are: ")

            epoch_sumlosses_tpl = sum_epoch_loss_global, sum_epoch_loss_sense, sum_epoch_loss_multisense
            epoch_numsteps_tpl = epoch_step, epoch_senselabeled_tokens, epoch_multisense_tokens
            Utils.record_statistics(epoch_sumlosses_tpl, epoch_numsteps_tpl, training_losses_lts)

            logging.info("Training - Correct predictions / Total predictions:")
            logging.info(correct_predictions_dict)
            # continue # skipping Validation in mini-experiments
            # Time to check the validation loss
            logging.info("After training " + str(epoch) + " epochs, the validation losses are:")
            valid_loss_globals, valid_loss_senses, multisenses_evaluation_loss = evaluation(valid_dataloader, valid_dataiter, model, verbose=False)
            validation_sumlosses = valid_loss_globals, valid_loss_senses, multisenses_evaluation_loss
            Utils.record_statistics(validation_sumlosses, (1,1,1), losses_lts=validation_losses_lts)
            epoch_valid_loss = valid_loss_globals + valid_loss_senses
            logging.info("Debug: epoch_valid_loss="+str(round(epoch_valid_loss,2)) + " Debug: best_valid_loss_globals="+str(round(best_valid_loss_globals,2)))

            # if exp(valid_loss_globals) > exp(best_valid_loss_globals) + 0.1: # if _new_ Valid PPL worse than _best_ by >0.1
            epoch_loss_globals = sum_epoch_loss_global / epoch_step #  for mini-experiments: globals' Train PPL computation
            if exp(epoch_loss_globals) < 3 : # for mini-experiments & debugging
                if not with_freezing:
                    # previous validation was better. Now we must early-stop
                    logging.info("Early stopping. Latest validation PPL=" + str(round(exp(epoch_valid_loss),2))
                                 + " ; best validation PPL="+str(round(exp(best_valid_loss_globals),2)))
                    flag_earlystop = True
                else:
                    if not after_freezing_flag:
                        # we are predicting first the standard LM, and then the senses. Freeze (1), activate (2).
                        logging.info("New validation worse than previous one. " +
                                     "Freezing the weights in the standard LM, activating senses' prediction.")
                        for (name, p) in model_forParameters.named_parameters():    # (1)
                            if ("main_rnn" in name) or ("X" in name) or ("linear2global" in name):
                                p.requires_grad=False
                        optimizers.append(torch.optim.Adam(model.parameters(), lr=learning_rate)) # [p for p in model.parameters() if p.requires_grad]
                        model_forParameters.predict_senses = True  # (2)
                        after_freezing_flag = True

            best_valid_loss_globals = min(best_valid_loss_globals, epoch_valid_loss)
            if epoch == num_epochs:
                write_doc_logging(train_dataloader, model, model_forParameters, learning_rate, num_epochs)

            if flag_earlystop:
                break

    except KeyboardInterrupt:
        logging.info("Training loop interrupted manually by keyboard")
    # save model
    torch.save(model, os.path.join(F.FOLDER_GNN, hyperparams_str +
                                   'step_' + str(overall_step) + '.rgcnmodel'))

    logging.info("Saving losses.")
    np.save(hyperparams_str + '_' + Utils.TRAINING + '_' + F.LOSSES_FILEEND, np.array(training_losses_lts))
    np.save(hyperparams_str + '_' + Utils.VALIDATION + '_' + F.LOSSES_FILEEND, np.array(validation_losses_lts))




################

def evaluation(evaluation_dataloader, evaluation_dataiter, model, verbose):
    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    including_senses = model_forParameters.predict_senses
    multisense_globals_set = set(AD.get_multisense_globals_indices())

    model.eval()  # do not train the model now
    sum_eval_loss_globals = 0
    sum_eval_loss_sense = 0
    sum_eval_loss_multisense = 0
    eval_correct_predictions_dict = {'correct_g':0,
                                        'top_k_g':0,
                                        'tot_g':0,
                                        'correct_all_s':0,
                                        'top_k_all_s':0,
                                        'tot_all_s':0,
                                        'correct_multi_s':0,
                                        'top_k_multi_s':0,
                                        'tot_multi_s':0
                                        }

    evaluation_step = 0
    evaluation_senselabeled_tokens = 0
    evaluation_multisense_tokens = 0
    logging_step = 500

    with torch.no_grad():  # Deactivates the autograd engine entirely to save some memory
        for b_idx in range(len(evaluation_dataloader)-1):
            batch_input, batch_labels = evaluation_dataiter.__next__()
            batch_input = batch_input.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            (losses_tpl, num_sense_instances_tpl) = compute_model_loss(model, batch_input, batch_labels, eval_correct_predictions_dict,
                                                          multisense_globals_set, verbose=verbose)
            loss_globals, loss_sense, loss_multisense = losses_tpl
            num_batch_sense_tokens, num_batch_multisense_tokens = num_sense_instances_tpl
            sum_eval_loss_globals = sum_eval_loss_globals + loss_globals.item()

            if including_senses:
                sum_eval_loss_sense = sum_eval_loss_sense + loss_sense.item() * num_batch_sense_tokens
                evaluation_senselabeled_tokens = evaluation_senselabeled_tokens + num_batch_sense_tokens
                sum_eval_loss_multisense = sum_eval_loss_multisense + loss_multisense.item() * num_batch_multisense_tokens
                evaluation_multisense_tokens = evaluation_multisense_tokens + num_batch_multisense_tokens

            evaluation_step = evaluation_step + 1
            if evaluation_step % logging_step == 0:
                logging.info("Evaluation step n. " + str(evaluation_step))
                gc.collect()

    globals_evaluation_loss = sum_eval_loss_globals / evaluation_step
    if including_senses:
        senses_evaluation_loss = sum_eval_loss_sense / evaluation_senselabeled_tokens
        multisenses_evaluation_loss = sum_eval_loss_multisense / evaluation_multisense_tokens
    else:
        senses_evaluation_loss = 0
        multisenses_evaluation_loss = 0

    logging.info("Validation - Correct predictions / Total predictions:")
    logging.info(eval_correct_predictions_dict)

    model.train()  # training can resume

    return globals_evaluation_loss, senses_evaluation_loss, multisenses_evaluation_loss



