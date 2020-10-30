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
from time import time
from VocabularyAndEmbeddings.ComputeEmbeddings import Method
from NN.Loss import write_doc_logging, compute_model_loss
from Utils import DEVICE
import NN.DataLoading as DL
import NN.Models.RNNs as RNNs
from itertools import cycle
import gc
from math import exp
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
from torch.nn.parameter import Parameter
from enum import Enum
import NN.Models.SelectK as SelectK
import NN.Models.SenseContextAverage as SC

################

class ModelType(Enum):
    RNN = "RNN"
    SELECTK = "SelectK"
    SC = "Sense Context"


def load_model_from_file(slc_or_text, inputdata_folder, graph_dataobj):
    saved_model_path = os.path.join(F.FOLDER_NN, F.SAVED_MODEL_NAME)
    model = torch.load(saved_model_path).module if torch.cuda.is_available() \
        else torch.load(saved_model_path, map_location=torch.device('cpu')).module # unwrapping DataParallel
    logging.info("Loading the model found at: " + str(saved_model_path))

    if slc_or_text:
    # we must replace the linear FF-NNs that go from LastLayerDim (eg. 512) to the vocabulary,
        last_idx_senses = graph_dataobj.node_types.tolist().index(1)
        num_senses = last_idx_senses + 1
        last_idx_globals = graph_dataobj.node_types.tolist().index(2)
        num_globals = last_idx_globals - last_idx_senses

        model.linear2global = torch.nn.Linear(in_features=model.hidden_size // 2, out_features=num_globals, bias=True)
        model.linear2senses = torch.nn.Linear(in_features=model.hidden_size // 2, out_features=num_senses, bias=True)

    # and the matrix E of embeddings too.
        E_embeddings = DG.load_word_embeddings(inputdata_folder)
        model.E = Parameter(E_embeddings.clone().detach(), requires_grad=True)

    return model

################

def setup_train(slc_or_text_corpus, model_type, include_globalnode_input, load_saved_model,
                batch_size=32, sequence_length=35,
                method=CE.Method.FASTTEXT, grapharea_size=32):

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
    # vocabulary_wordList = vocabulary_df['word'].to_list().copy()

    vocabulary_numSensesList = vocabulary_df['num_senses'].to_list().copy()
    if all([num_senses == -1 for num_senses in vocabulary_numSensesList]):
        vocabulary_df = AD.compute_globals_numsenses(graph_dataobj, grapharea_matrix, grapharea_size, slc_or_text_corpus)

    # -------------------- Loading / creating the model --------------------
    torch.manual_seed(1) # for reproducibility while conducting mini-experiments
    if torch.cuda.is_available():
         torch.cuda.manual_seed_all(1)
    if load_saved_model:
        model = load_model_from_file(slc_or_text_corpus, inputdata_folder, graph_dataobj)
    else:
        if model_type==ModelType.RNN:
            model = RNNs.RNN(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                               embeddings_matrix, include_globalnode_input,
                               batch_size, n_layers=3, n_hid_units=1024)
        elif model_type==ModelType.SELECTK:
            model = SelectK.SelectK(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                                    include_globalnode_input, batch_size, n_layers=3, n_hid_units=1024, K=5)
        elif model_type==ModelType.SC:
            model = SC.SenseContextAverage(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                                           embeddings_matrix, include_globalnode_input, batch_size, n_layers=3,
                                           n_hid_units=1024, K=1, num_C=10)
        else:
            raise Exception ("Model type specification incorrect")
        # model = Senses.ContextSim(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
        #                            batch_size, seq_len=sequence_length, n_layers=3, n_hid_units=1024, k=10, c=10)

    # -------------------- Moving objects on GPU --------------------
    logging.info("Graph-data object loaded, model initialized. Moving them to GPU device(s) if present.")

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1: # and allow_dataparallel:
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
    train_corpus_fpath = os.path.join(F.FOLDER_TEXT_CORPUSES, subfolder, F.FOLDER_TRAIN)
    valid_corpus_fpath = os.path.join(F.FOLDER_TEXT_CORPUSES, subfolder, F.FOLDER_VALIDATION)
    train_dataset = DL.TextDataset(slc_or_text_corpus, train_corpus_fpath, senseindices_db_c, vocab_h5, model_forDataLoading,
                                   grapharea_matrix, grapharea_size, graph_dataobj)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size * sequence_length,
                                                   num_workers=0, collate_fn=bptt_collator)

    valid_dataset = DL.TextDataset(slc_or_text_corpus, valid_corpus_fpath, senseindices_db_c, vocab_h5, model_forDataLoading,
                                   grapharea_matrix, grapharea_size, graph_dataobj)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size * sequence_length,
                                                   num_workers=0, collate_fn=bptt_collator)

    return model, train_dataloader, valid_dataloader


################


################
def run_train(model, train_dataloader, valid_dataloader, learning_rate, num_epochs, predict_senses, with_freezing):

    # -------------------- Setup; parameters and utilities --------------------
    Utils.init_logging('Training' + Utils.get_timestamp_month_to_min() + '.log', loglevel=logging.INFO)
    slc_or_text = train_dataloader.dataset.sensecorpus_or_text

    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate)]

    model.train()
    training_losses_lts = []
    validation_losses_lts = []
    multisense_globals_set = set(AD.get_multisense_globals_indices(slc_or_text))

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    model_forParameters.predict_senses = predict_senses
    hyperparams_str = write_doc_logging(train_dataloader, model, model_forParameters, learning_rate, num_epochs)

    steps_logging = 50
    overall_step = 0
    starting_time = time()
    best_valid_loss_globals = inf
    best_valid_loss_senses = inf
    previous_valid_loss_senses = inf
    freezing_epoch = None
    after_freezing_flag = False
    if with_freezing:
        model_forParameters.predict_senses = False
    weights_before_freezing_check_ls = []
    parameters_to_check_names_ls = []

    # debug
    # torch.autograd.set_detect_anomaly(True)

    train_dataiter = iter(cycle(train_dataloader))
    valid_dataiter = iter(cycle(valid_dataloader))

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

            flag_earlystop = False

            # -------------------- Running on the validation set --------------------
            if epoch>1:
                logging.info("After training " + str(epoch-1) + " epochs, validation:")
                valid_loss_globals, valid_loss_senses, multisenses_evaluation_loss = evaluation(valid_dataloader,
                                                                                                valid_dataiter,
                                                                                                model, slc_or_text,
                                                                                                verbose=False)
                validation_sumlosses = valid_loss_globals, valid_loss_senses, multisenses_evaluation_loss
                Utils.record_statistics(validation_sumlosses, (1, 1, 1), losses_lts=validation_losses_lts)

                # -------------------- Check the validation loss & the need for freezing / early stopping --------------------
                if exp(valid_loss_globals) > exp(best_valid_loss_globals) + 0.1 and (epoch > 2):

                    if exp(valid_loss_globals) > exp(best_valid_loss_globals):
                        torch.save(model, os.path.join(F.FOLDER_NN, hyperparams_str + '_best_validation.model'))
                    if not with_freezing:
                        # previous validation was better. Now we must early-stop
                        logging.info("Early stopping. Latest validation PPL=" + str(round(exp(valid_loss_globals), 2))
                                     + " ; best validation PPL=" + str(round(exp(best_valid_loss_globals), 2)))
                        flag_earlystop = True
                    else:
                        if not after_freezing_flag:
                            # we are predicting first the standard LM, and then the senses. Freeze (1), activate (2).
                            logging.info("New validation worse than previous one. " +
                                         "Freezing the weights in the standard LM, activating senses' prediction.")
                            logging.info("Status of parameters before freezing:")
                            for (name, p) in model_forParameters.named_parameters():  # (1)
                                logging.info("Parameter=" + str(name), " ; requires_grad=" + str(p.requires_grad))
                                parameters_to_check_names_ls.append(name)
                                weights_before_freezing_check_ls.append(p.clone().detach())

                                if ("main_rnn" in name) or ("E" in name) or ("X" in name) or ("linear2global" in name):
                                    p.requires_grad = False
                                    p.grad = p.grad * 0
                            optimizers.append(torch.optim.Adam(model.parameters(),
                                                               lr=learning_rate))  # [p for p in model.parameters() if p.requires_grad]
                            optimizer = optimizers[-1]  # pick the most recently created optimizer
                            model_forParameters.predict_senses = True  # (2)
                            after_freezing_flag = True
                            logging.info("Status of parameters after freezing:")
                            for (name, p) in model_forParameters.named_parameters():
                                logging.info("Parameter=" + str(name), " ; requires_grad=" + str(p.requires_grad))
                            freezing_epoch = epoch
                if after_freezing_flag:
                    if (exp(valid_loss_senses) > exp(previous_valid_loss_senses) + 1) and epoch > freezing_epoch + 2:
                        logging.info("Early stopping on senses.")
                        flag_earlystop = True

                best_valid_loss_globals = min(best_valid_loss_globals, valid_loss_globals)
                previous_valid_loss_senses = valid_loss_senses
                best_valid_loss_senses = min(best_valid_loss_senses, valid_loss_senses)

                if epoch == num_epochs:
                    write_doc_logging(train_dataloader, model, model_forParameters, learning_rate, num_epochs)

                if flag_earlystop:
                    break

            # -------------------- The training loop over the batches --------------------
            logging.info("\nEpoch n." + str(epoch) + ":")
            for b_idx in range(len(train_dataloader)-1):
                t0 = time()
                batch_input, batch_labels = train_dataiter.__next__()
                batch_input = batch_input.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

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

    # --------------------- Saving model and losses --------------------
    torch.save(model, os.path.join(F.FOLDER_NN, hyperparams_str + '_end.model'))
    np.save(hyperparams_str + '_' + Utils.TRAINING + '_' + F.LOSSES_FILEEND, np.array(training_losses_lts))
    np.save(hyperparams_str + '_' + Utils.VALIDATION + '_' + F.LOSSES_FILEEND, np.array(validation_losses_lts))

    # --------------------- Printing validation predictions & errors ---------------------
    evaluation(valid_dataloader, valid_dataiter, model, verbose=True, slc_or_text=slc_or_text)




################

def evaluation(evaluation_dataloader, evaluation_dataiter, model, slc_or_text, verbose):
    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    including_senses = model_forParameters.predict_senses
    multisense_globals_set = set(AD.get_multisense_globals_indices(slc_or_text))

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
                                                                       multisense_globals_set, slc_or_text, verbose=verbose)
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

    logging.info("Validation - Correct predictions / Total predictions:\n" + str(eval_correct_predictions_dict))

    model.train()  # training can resume

    return globals_evaluation_loss, senses_evaluation_loss, multisenses_evaluation_loss



