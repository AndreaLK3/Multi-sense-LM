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
import Graph.Adjacencies as AD
import GNN.DataLoading as DL
import GNN.ExplorePredictions as EP
import GNN.Models.RNNs as RNNs
import GNN.Models.Senses as SensesNets
from itertools import cycle
import gc

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


def update_predictions_history_dict(correct_preds_dict, predictions_globals, predictions_senses, batch_labels):

    k = 10
    batch_labels_globals = batch_labels[:,0]
    batch_labels_senses = batch_labels[:,1]

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
        correct_preds_dict['correct_s'] = \
            correct_preds_dict['correct_s'] + torch.sum(indices_s[:, 0] == batch_labels_senses).item()
        correct_preds_dict['tot_s'] = correct_preds_dict['tot_s'] + \
                                      (batch_labels.t()[1][batch_labels.t()[1] != -1].shape[0])

        top_k_predictions_s = indices_s[:, 0:k]
        batch_counter_top_k_s = 0
        for i in range(len(batch_labels_senses)):
            label_s = batch_labels_senses[i]
            if label_s in top_k_predictions_s[i]:
                batch_counter_top_k_s = batch_counter_top_k_s + 1
        correct_preds_dict['top_k_s'] = correct_preds_dict['top_k_s'] + batch_counter_top_k_s

    logging.debug("updated_predictions_history_dict = " +str(correct_preds_dict))
    return


def compute_model_loss(model, batch_input, batch_labels, correct_preds_dict, verbose=False):
    predictions_globals, predictions_senses = model(batch_input)

    batch_labels_t = (batch_labels).clone().t().to(DEVICE)
    batch_labels_globals = batch_labels_t[0]
    batch_labels_all_senses = batch_labels_t[1]

    # compute the loss for the batch
    loss_global = tfunc.nll_loss(predictions_globals, batch_labels_globals)

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    if model_forParameters.predict_senses:
        loss_all_senses = tfunc.nll_loss(predictions_senses, batch_labels_all_senses, ignore_index=-1)
    else:
        loss_all_senses = torch.tensor(0)

    # Added to measure the senses' task, given that we can not rely on the senses' PPL for SelectK
    update_predictions_history_dict(correct_preds_dict, predictions_globals, predictions_senses, batch_labels)

    # debug: check the solutions and predictions. Is there anything the model is unable to predict?
    if verbose:
        logging.info("*******\ncompute_model_loss > verbose logging of batch")
        EP.log_batch(batch_labels, predictions_globals, predictions_senses, 5)

    return loss_global, loss_all_senses


################

def training_setup(slc_or_text_corpus, include_globalnode_input, include_sensenode_input, predict_senses,
                   method, grapharea_size, batch_size, sequence_length, allow_dataparallel=True):
    graph_dataobj = DG.get_graph_dataobject(new=False, method=method, slc_corpus=slc_or_text_corpus).to(DEVICE)
    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, grapharea_size, hops_in_area=1)

    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE)
    vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')
    globals_vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')

    if all([num_senses==-1 for num_senses in globals_vocabulary_df['num_senses'].tolist()]):
        AD.compute_globals_numsenses(graph_dataobj, grapharea_matrix, grapharea_size)

    # The original GRU architecture has been updated into the GRUbase2 model in Senses - I just have to specify that predict_senses=False
    # torch.manual_seed(1) # for reproducibility while conducting mini-experiments
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(1)
    model = RNNs.RNN("LSTM", graph_dataobj, grapharea_size, grapharea_matrix, globals_vocabulary_df,
                      include_globalnode_input, include_sensenode_input, predict_senses,
                      batch_size=batch_size, n_layers=3, n_hid_units=1150)
    # model = SensesNets.SelectK(graph_dataobj, grapharea_size, grapharea_matrix, 10, globals_vocabulary_wordList,
    #                            include_globalnode_input, include_sensenode_input, predict_senses,
    #                            batch_size, n_layers=3, n_hid_units=1024)

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
def training_loop(model, learning_rate, train_dataloader, valid_dataloader, num_epochs):

    Utils.init_logging('Training' + Utils.get_timestamp_month_to_min() + '.log', loglevel=logging.INFO)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # weight_decay=0.0005

    model.train()
    training_losses_lts = [] # mutated into a lts, with (global_loss, sense_loss)
    validation_losses_lts = []

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    hyperparams_str = write_doc_logging(train_dataloader, model, model_forParameters, learning_rate, num_epochs)

    steps_logging = 100
    overall_step = 0
    starting_time = time()
    previous_valid_loss = inf
    flag_firstvalidationhigher = False

    try: # to catch KeyboardInterrupt-s and still save the training & validation losses
        ########## The training loop ##########
        train_dataiter = iter(cycle(train_dataloader))
        valid_dataiter = iter(cycle(valid_dataloader))
        for epoch in range(1,num_epochs+1):
            logging.info("\nTraining epoch n."+str(epoch) + ":")
            if epoch == 300:
                logging.info("debug of correct predictions")

            sum_epoch_loss_global = 0
            sum_epoch_loss_sense = 0
            epoch_step = 0
            epoch_senselabeled_tokens = 0
            correct_predictions_dict = {'correct_g':0,
                                        'top_k_g':0,
                                        'tot_g':0,
                                        'correct_s':0,
                                        'top_k_s':0,
                                        'tot_s':0}
            verbose = True if (epoch==num_epochs) or (epoch% 100==0) else False # - log prediction output

            flag_earlystop = False

            for b_idx in range(len(train_dataloader)):
                t0=time()
                batch_input, batch_labels = train_dataiter.__next__()
                batch_input = batch_input.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                # starting operations on one batch
                optimizer.zero_grad()

                # compute loss for the batch
                loss_global, loss_sense = compute_model_loss(model, batch_input, batch_labels, correct_predictions_dict, verbose)
                #logging.info("Batch n.: " + str(b_idx) + " loss_global = " + str(loss_global.item()))
                # running sum of the training loss in the log segment
                sum_epoch_loss_global = sum_epoch_loss_global + loss_global.item()
                if model_forParameters.predict_senses:
                    # the senses are weighted depending on the number of sense labels, so they are not skewed from no-labels
                    batch_sense_tokens = (batch_labels.t()[1][batch_labels.t()[1]!=-1].shape[0])
                    sum_epoch_loss_sense = sum_epoch_loss_sense + loss_sense.item() * batch_sense_tokens
                    epoch_senselabeled_tokens = epoch_senselabeled_tokens + batch_sense_tokens
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
            # except StopIteration: the DataLoader naturally catches StopIteration
                # end of an epoch.

            logging.info("-----\n Training, end of epoch " + str(epoch) + ". Global step n." + str(overall_step) +
                         ". Time = " + str(round(time() - starting_time, 2)) + ". The training losses are: ")
            Utils.record_statistics(sum_epoch_loss_global, sum_epoch_loss_sense, epoch_step,
                                    max(1,epoch_senselabeled_tokens), training_losses_lts)

            logging.info("Training - Correct predictions / Total predictions:")
            logging.info(correct_predictions_dict)
            continue # skipping Validation in mini-experiments
            # Time to check the validation loss
            logging.info("After training " + str(epoch) + " epochs, the validation losses are:")
            valid_loss_globals, valid_loss_senses = evaluation(valid_dataloader, valid_dataiter, model)
            #validation_losses_lts.append((valid_loss_globals, valid_loss_senses))
            Utils.record_statistics(valid_loss_globals, valid_loss_senses, 1, 1, losses_lts=validation_losses_lts)
            epoch_valid_loss = valid_loss_globals + valid_loss_senses

            if epoch_valid_loss < previous_valid_loss:
                pass # we can save the model often here if we so wish
            if epoch_valid_loss > previous_valid_loss + 10:
                if not flag_firstvalidationhigher:
                    flag_firstvalidationhigher = True
                    logging.info("Validation loss worse than previous one. First occurrence.")
                else: # previous best validation was better once already. Now we must early-stop
                    logging.info("Early stopping")
                    flag_earlystop = True
            previous_valid_loss = epoch_valid_loss

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

def evaluation(evaluation_dataloader, evaluation_dataiter, model):
    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    including_senses = model_forParameters.predict_senses

    model.eval()  # do not train the model now
    sum_eval_loss_globals = 0
    sum_eval_loss_sense = 0
    eval_correct_predictions_dict ={'correct_g':0,
                                    'top_k_g':0,
                                    'tot_g':0,
                                    'correct_s':0,
                                    'top_k_s':0,
                                    'tot_s':0}

    evaluation_step = 0
    evaluation_senselabeled_tokens = 0
    logging_step = 500

    with torch.no_grad(): # Deactivates the autograd engine entirely to save some memory
        for b_idx in range(len(evaluation_dataloader)):
            batch_input, batch_labels = evaluation_dataiter.__next__()
            batch_input = batch_input.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            loss_globals, loss_sense = compute_model_loss(model, batch_input, batch_labels, eval_correct_predictions_dict, verbose=False)
            sum_eval_loss_globals = sum_eval_loss_globals + loss_globals.item()

            if including_senses:
                num_batch_sense_tokens = batch_labels.t()[1][batch_labels.t()[1]!=-1].shape[0]
                sum_eval_loss_sense = sum_eval_loss_sense + loss_sense.item() * num_batch_sense_tokens
                evaluation_senselabeled_tokens = evaluation_senselabeled_tokens + num_batch_sense_tokens


            evaluation_step = evaluation_step + 1
            if evaluation_step % logging_step == 0:
                logging.info("Evaluation step n. " + str(evaluation_step))
                gc.collect()

    globals_evaluation_loss = sum_eval_loss_globals / evaluation_step
    if including_senses:
        senses_evaluation_loss = sum_eval_loss_sense / evaluation_senselabeled_tokens
    else:
        senses_evaluation_loss = 0

    logging.info("Validation - Correct predictions / Total predictions:")
    logging.info(eval_correct_predictions_dict)

    model.train()  # training can resume

    return globals_evaluation_loss, senses_evaluation_loss



