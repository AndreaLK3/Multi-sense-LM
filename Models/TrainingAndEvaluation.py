import Graph.PolysemousWords
import Models.Loss
import Utils
import logging
import torch
import Graph.Adjacencies as AD
import Models.Loss as Loss
from time import time
from math import inf, exp
import itertools
import os
import Filesystem as F
from Utils import DEVICE
import gc
import VocabularyAndEmbeddings.ComputeEmbeddings as CE

# Auxiliary function: initialize a dictionary that registers the accuracy
def init_accuracy_dict(polysense_thresholds):
    return {'correct_g': 0,
     'tot_g': 0,
     'correct_all_s': 0,
     'tot_all_s': 0,
     'correct_poly_s': {}.fromkeys(polysense_thresholds, 0),
     'tot_poly_s': {}.fromkeys(polysense_thresholds, 0)
     }

################
def run_train(model, train_dataloader, valid_dataloader, learning_rate, num_epochs, predict_senses=True,
              vocab_sources_ls=(F.WT2, F.SEMCOR), sp_method=Utils.SpMethod.FASTTEXT):

    # -------------------- Step 1: Setup model --------------------
    slc_or_text = train_dataloader.dataset.sensecorpus_or_text
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    polysense_thresholds = (2,3,5,10,30)
    polysense_globals_dict= Graph.PolysemousWords.get_polysenseglobals_dict(vocab_sources_ls, sp_method, thresholds=polysense_thresholds)

    model.predict_senses = predict_senses

    model_fname = F.get_model_name(model, args=None)
    Utils.init_logging("Training_" + model_fname.replace(".pt", "") + ".log")
    try:
        logging.info("Using learning_rate=" + str(learning_rate))
        logging.info("K=" + str(model.K))
        logging.info("C=" + str(model.C))
        logging.info("context_method=" + str(model.context_method))
    except Exception:
        pass # no further hyperparameters were specified
    Loss.write_doc_logging(model)

    # -------------------- Step 2: Setup flags --------------------
    steps_logging = 500
    overall_step = 0
    starting_time = time()
    best_valid_loss_globals = inf
    best_valid_loss_senses = inf
    best_valid_accuracy_senses = 0

    # torch.autograd.set_detect_anomaly(True) # for debug
    train_dataiter = iter(itertools.cycle(train_dataloader))
    valid_dataiter = iter(itertools.cycle(valid_dataloader))

    # -------------------- Step 3) The training loop, for each epoch --------------------
    try: # to catch KeyboardInterrupt-s and still save the model

        for epoch in range(1,num_epochs+1):

            # -------------------- Step 3a) Initialization --------------------
            sum_epoch_loss_global = 0
            sum_epoch_loss_senses = 0

            epoch_step = 1
            epoch_senselabeled_tokens = 0

            predictions_history_dict = init_accuracy_dict(polysense_thresholds)
            verbose = False # True if (epoch==num_epochs) or (epoch% 200==0) else False # deciding: log prediction output
            flag_earlystop = False

            # -------------------- Step 3b) Evaluation on the validation set --------------------
            if epoch>1:
                logging.info("After training " + str(epoch-1) + " epochs, validation:")
                valid_loss_globals, valid_loss_senses, valid_accuracy_senses = \
                    evaluation(valid_dataloader, valid_dataiter, model, verbose)

                # -------------- 3c) Check the validation accuracy & the need for early stopping --------------
                if valid_accuracy_senses <= best_valid_accuracy_senses and (epoch > 2) and (valid_accuracy_senses>0):
                    logging.info("Early stopping on senses' accuracy.")
                    flag_earlystop = True
                if valid_accuracy_senses == 0: # we are operating on globals only, i.e. WT-2
                    if valid_loss_globals > best_valid_loss_globals:
                        logging.info("Early stopping on globals' PPL")
                        flag_earlystop = True

                best_valid_loss_globals = min(best_valid_loss_globals, valid_loss_globals)
                best_valid_loss_senses = min(best_valid_loss_senses, valid_loss_senses)
                best_valid_accuracy_senses = max(best_valid_accuracy_senses, valid_accuracy_senses)

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
                loss_global, loss_sense = losses_tpl
                num_batch_sense_tokens = num_sense_instances_tpl

                # running sum of the training loss
                sum_epoch_loss_global = sum_epoch_loss_global + loss_global.item()
                if model.predict_senses:
                    sum_epoch_loss_senses = sum_epoch_loss_senses + loss_sense.item() * num_batch_sense_tokens
                    epoch_senselabeled_tokens = epoch_senselabeled_tokens + num_batch_sense_tokens
                    loss = loss_global + loss_sense
                else:
                    loss = loss_global

                if loss.requires_grad: # to cover the gold_lm + MFS case, that has no gradient
                    loss.backward()

                optimizer.step()
                overall_step = overall_step + 1
                epoch_step = epoch_step + 1

                if overall_step % steps_logging == 0:
                    logging.info("Global step=" + str(overall_step) + "\t ; Iteration time=" + str(round(time()-t0,5)))
                    gc.collect()
                # end of an epoch.

            # -------------------- 3e) Computing training losses for the epoch--------------------
            logging.info("-----\n Training, end of epoch " + str(epoch) + ". Global step n." + str(overall_step) +
                         ". Time = " + str(round(time() - starting_time, 2)))
            logging.info("Models - Correct predictions / Total predictions:\n" + str(predictions_history_dict))

            epoch_sumlosses_tpl = sum_epoch_loss_global, sum_epoch_loss_senses
            epoch_numsteps_tpl = epoch_step, epoch_senselabeled_tokens
            _senses_acc = Models.Loss.record_statistics(epoch_sumlosses_tpl, epoch_numsteps_tpl, predictions_history_dict)

    except KeyboardInterrupt:
        logging.info("Models loop interrupted manually by keyboard")

    # --------------------- 4) Saving model --------------------
    logging.info("Proceeding to save the model: " + model_fname + ", timestamp: " + Utils.get_timestamp_month_to_sec())
    torch.save(model, os.path.join(F.FOLDER_SAVEDMODELS, model_fname))

    return model


# ##########
# Auxiliary function: Evaluation on a given dataset, e.g. validation or test set
def evaluation(evaluation_dataloader, evaluation_dataiter, model, verbose, vocab_sources_ls=(F.WT2, F.SEMCOR),
               sp_method=Utils.SpMethod.FASTTEXT):
    including_senses = model.predict_senses

    polysense_thresholds = (2, 3, 5, 10, 30)
    polysense_globals_dict = Graph.PolysemousWords.get_polysenseglobals_dict(vocab_sources_ls, sp_method)

    model.eval()  # do not train the model now
    sum_eval_loss_globals = 0
    sum_eval_loss_senses = 0
    eval_correct_predictions_dict = init_accuracy_dict(polysense_thresholds)
    evaluation_step = 1
    evaluation_senselabeled_tokens = 0
    logging_step = 500

    with torch.no_grad():  # Deactivates the autograd engine entirely to save some memory
        for b_idx in range(len(evaluation_dataloader)-1):
            batch_input, batch_labels = evaluation_dataiter.__next__()
            batch_input = batch_input.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            (losses_tpl, num_sense_instances_tpl) \
                = Loss.compute_model_loss(model, batch_input, batch_labels, eval_correct_predictions_dict,
                                          polysense_globals_dict, vocab_sources_ls, sp_method, verbose=verbose)
            loss_globals, loss_senses = losses_tpl
            num_batch_sense_tokens = num_sense_instances_tpl
            sum_eval_loss_globals = sum_eval_loss_globals + loss_globals.item()

            if including_senses:
                sum_eval_loss_senses = sum_eval_loss_senses + loss_senses.item() * num_batch_sense_tokens
                evaluation_senselabeled_tokens = evaluation_senselabeled_tokens + num_batch_sense_tokens

            evaluation_step = evaluation_step + 1
            if evaluation_step % logging_step == 0:
                logging.info("Evaluation step n. " + str(evaluation_step))
                gc.collect()

    globals_evaluation_loss = sum_eval_loss_globals / evaluation_step
    if including_senses:
        senses_evaluation_loss = sum_eval_loss_senses / evaluation_senselabeled_tokens
    else:
        senses_evaluation_loss = 0

    logging.info("Evaluation - Correct predictions / Total predictions:\n" + str(eval_correct_predictions_dict))
    epoch_sumlosses_tpl = sum_eval_loss_globals, sum_eval_loss_senses
    epoch_numsteps_tpl = evaluation_step, evaluation_senselabeled_tokens
    senses_acc = Models.Loss.record_statistics(epoch_sumlosses_tpl, epoch_numsteps_tpl, eval_correct_predictions_dict)

    model.train()  # training can resume

    return globals_evaluation_loss, senses_evaluation_loss, senses_acc


