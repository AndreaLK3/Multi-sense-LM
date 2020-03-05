import torch
from torch_geometric.nn import RGCNConv
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tfunc
import torch.nn.modules.batchnorm as batchnorm
import Graph.DefineGraph as DG
from math import exp
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
import GNN.MyRGCN as MyRGCN


# Auxiliary function for compute_model_loss
def compute_sense_loss(predictions_senses, batch_labels_senses):
    batch_validsenses_predicted = []
    batch_validsenses_labels = []
    for i in range(batch_labels_senses.shape[0]):
        senselabel = batch_labels_senses[i]
        if senselabel != -1:
            batch_validsenses_labels.append(senselabel.item())
            batch_validsenses_predicted.append(predictions_senses[i])
    if len(batch_validsenses_labels) >= 1:
        loss_sense = tfunc.nll_loss(torch.stack(batch_validsenses_predicted).to(DEVICE),
                                    torch.tensor(batch_validsenses_labels, dtype=torch.int64).to(DEVICE))
    else:
        loss_sense = torch.tensor(0).to(DEVICE)
    return loss_sense

################

def compute_model_loss(model,batch_input, batch_labels, verbose=False):
    predictions_globals, predictions_senses = model(batch_input)

    batch_labels_t = (batch_labels).clone().t().to(DEVICE)
    batch_labels_globals = batch_labels_t[0]
    batch_labels_senses = batch_labels_t[1]

    # compute the loss for the batch
    loss_global = tfunc.nll_loss(predictions_globals, batch_labels_globals)
    loss_sense = compute_sense_loss(predictions_senses, batch_labels_senses)

    # debug: check the solutions and predictions. Is there anything the model is unable to predict?
    if verbose:
        EP.log_batch(batch_labels, predictions_globals, predictions_senses, 5)

    return loss_global, loss_sense


################

def record_statistics(sum_epoch_loss_global, sum_epoch_loss_sense, epoch_step, num_steps_withsense, losses_lts):
    epoch_loss_globals = sum_epoch_loss_global / epoch_step
    epoch_loss_senses = sum_epoch_loss_sense / num_steps_withsense
    epoch_loss = epoch_loss_globals + epoch_loss_senses
    logging.info("Losses: " + " Globals loss=" + str(round(epoch_loss_globals,3)) +
                               " \tSense loss=" + str(round(epoch_loss_senses,3)) +
                               " \tTotal loss=" + str(round(epoch_loss,3)) )
    logging.info("Perplexity: " + " Globals perplexity=" + str(round(exp(epoch_loss_globals),3)) +
                 " \tSense perplexity=" + str(round(exp(epoch_loss_senses),3)) + "\n-------")
    losses_lts.append((epoch_loss_globals, epoch_loss_senses))


################

def train(grapharea_size=32, size_batch=None, sequence_length=8, learning_rate=0.001, num_epochs=100):
    Utils.init_logging('Training'+Utils.get_timestamp_month_to_min()+'.log')
    graph_dataobj = DG.get_graph_dataobject(new=False)
    logging.info(graph_dataobj)
    model = MyRGCN.GRU_RGCN(graph_dataobj, grapharea_size)
    logging.info("Graph-data object loaded, model initialized. Moving them to GPU device(s) if present.")
    graph_dataobj.to(DEVICE)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
         logging.info("Using "  +str(n_gpu) + " GPUs")
         model = torch.nn.DataParallel(model)
         model_forDataLoading = model.module
         batch_size = n_gpu if size_batch is None else size_batch # if not specified, default batch_size = n. GPUs
    else:
        model_forDataLoading = model
        batch_size = 1 if size_batch is None else size_batch
    model.to(DEVICE)

    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, grapharea_size)

    senseindices_db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()

    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE)
    vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #  weight_decay=0.0005

    model.train()
    training_losses_lts = [] # mutated into a lts, with (global_loss, sense_loss)
    validation_losses_lts = []
    steps_logging = 5000 // sequence_length
    hyperparams_str = 'model' + str(type(model).__name__) \
                      + '_batch' + str(sequence_length) \
                      + '_area' + str(grapharea_size)\
                      + '_lr' + str(learning_rate) \
                      + '_epochs' + str(num_epochs)
    logging.info("Parameters:")
    parameters_list = [(name, param.shape, param.requires_grad) for (name, param) in model.named_parameters()]
    logging.info(parameters_list)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info("Number of trainable parameters=" + str(params))

    overall_step = 0
    starting_time = time()
    previous_valid_loss = inf
    flag_firstvalidationhigher = False

    bptt_collator = DL.BPTTBatchCollator(grapharea_size, sequence_length)

    try: # to catch KeyboardInterrupt-s and still save the training & validation losses
        ########## The training loop ##########

        for epoch in range(1,num_epochs+1):
            logging.info("\nTraining epoch n."+str(epoch) + ":")
            train_dataset = DL.TextDataset('training', senseindices_db_c, vocab_h5, model_forDataLoading,
                                           grapharea_matrix, grapharea_size, graph_dataobj)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size * sequence_length,
                                                           num_workers=0, collate_fn=bptt_collator)

            sum_epoch_loss_global = 0
            sum_epoch_loss_sense = 0
            epoch_step = 0
            epoch_senselabeled_tokens = 0
            verbose = True if epoch==num_epochs else False # log output if in last epoch

            flag_earlystop = False

            for batch_input, batch_labels in train_dataloader: # tuple of 2 lists
                #logging.info("Step " + str(epoch_step))
                batch_input = batch_input.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                # starting operations on one batch
                optimizer.zero_grad()
                t0 = time()

                # compute loss for the batch
                loss_global, loss_sense = compute_model_loss(model, batch_input, batch_labels, verbose)

                # running sum of the training loss in the log segment
                sum_epoch_loss_global = sum_epoch_loss_global + loss_global.item()
                # the senses are weighted depending on the number of sense labels, so they are not skewed from no-labels
                batch_sense_tokens = (batch_labels.t()[1][batch_labels.t()[1]!=-1].shape[0])
                sum_epoch_loss_sense = sum_epoch_loss_sense + loss_sense.item() * batch_sense_tokens

                loss = loss_global + loss_sense
                loss.backward()
                #In the current version, we allow for defs and examples to be moved
                #last_embedding_to_update = model_forDataLoading.last_idx_senses + model_forDataLoading.last_idx_globals
                #model_forDataLoading.X.grad.data[last_embedding_to_update:,:].fill_(0) # defs and examples should not change
                optimizer.step()

                overall_step = overall_step + 1
                epoch_step = epoch_step + 1
                epoch_senselabeled_tokens = epoch_senselabeled_tokens + batch_sense_tokens


                if overall_step % steps_logging == 0:
                    logging.info("Global step=" + str(overall_step) + "\t ; Iteration time=" + str(round(time()-t0,5)))

                #Utils.log_chronometer([t0, time()])

            # except StopIteration: the DataLoader naturally catches StopIteration
                # end of an epoch.
            logging.info("-----\n Training, end of epoch " + str(epoch) + ". Global step n." + str(overall_step) +
                         ". Time = " + str(round(time() - starting_time, 2)) + ". The training losses are: ")
            record_statistics(sum_epoch_loss_global, sum_epoch_loss_sense, epoch_step, epoch_senselabeled_tokens, training_losses_lts)

            # Time to check the validation loss
            valid_dataset = DL.TextDataset('validation', senseindices_db_c, vocab_h5, model_forDataLoading,
                                           grapharea_matrix, grapharea_size, graph_dataobj)
            valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=sequence_length, num_workers=0,
                                                           collate_fn=bptt_collator)
            valid_loss_globals, valid_loss_senses = validation(valid_dataloader, model)
            validation_losses_lts.append((valid_loss_globals, valid_loss_senses))
            logging.info("-----\n After training " + str(epoch)+  " epochs, the validation losses are:")
            record_statistics(valid_loss_globals, valid_loss_senses, 1,1, losses_lts=validation_losses_lts)
            epoch_valid_loss = valid_loss_globals + valid_loss_senses

            #if epoch_valid_loss < previous_valid_loss:
                # save model

            if epoch_valid_loss > previous_valid_loss + 0.01 :
                if not flag_firstvalidationhigher:
                    flag_firstvalidationhigher = True
                else: # already did first offence. Must early-stop
                    logging.info("Early stopping")
                    flag_earlystop = True
            previous_valid_loss = epoch_valid_loss

            if flag_earlystop:
                break

    except KeyboardInterrupt:
        logging.info("Training loop interrupted manually by keyboard")

    logging.info("Saving losses and RGCN model.")
    np.save(hyperparams_str + '_' + Utils.TRAINING + '_' + F.LOSSES_FILEEND, np.array(training_losses_lts))
    np.save(hyperparams_str + '_' + Utils.VALIDATION + '_' + F.LOSSES_FILEEND, np.array(validation_losses_lts))
    torch.save(model, os.path.join(F.FOLDER_GNN, hyperparams_str +
                                   'step_' + str(overall_step) + '.rgcnmodel'))



################

def validation(valid_dataloader, model):

    model.eval()  # do not train the model now
    sum_valid_loss_globals = 0
    sum_valid_loss_sense = 0

    validation_step = 0
    validation_senselabeled_tokens = 0
    logging_step = 1000

    with torch.no_grad(): # Deactivates the autograd engine entirely to save some memory
        for batch_input, batch_labels in valid_dataloader:
            valid_loss_globals, valid_loss_sense = compute_model_loss(model, batch_input, batch_labels, verbose=False)
            sum_valid_loss_globals = sum_valid_loss_globals + valid_loss_globals.item()
            num_batch_sense_tokens = batch_labels.t()[1][batch_labels.t()[1]!=-1].shape[0]
            sum_valid_loss_sense = sum_valid_loss_sense + valid_loss_sense.item() * num_batch_sense_tokens

            validation_senselabeled_tokens = validation_senselabeled_tokens + num_batch_sense_tokens
            validation_step = validation_step + 1
            if validation_step % logging_step == 0:
                logging.info("Validation step n. " + str(validation_step))

    globals_validation_loss = sum_valid_loss_globals / validation_step
    senses_validation_loss = sum_valid_loss_sense / validation_senselabeled_tokens

    model.train()  # training can resume

    return globals_validation_loss, senses_validation_loss



