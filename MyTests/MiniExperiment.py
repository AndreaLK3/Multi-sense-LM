# This file should mostly copy the NN/Training.py file, although it is not used to set up standard experiments,
# but instead small mini-experiments (overfitting on a fragment of the training set).
# We operate on separate mini-corpuses, and print the input processed by the RNNs forward() call.

import torch
import Utils
import Filesystem as F
import logging
import os
import Graph.Adjacencies as AD
from NN.Training import load_model_from_file, ModelType, get_objects, get_dataloaders, create_model, evaluation
from NN.Loss import write_doc_logging, compute_model_loss
from Utils import DEVICE
from itertools import cycle
import gc
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
def setup_train(slc_or_text_corpus, model_type, K=0, C=0, context_method=None,
                dim_qkv=300,
                include_globalnode_input=0, load_saved_model=False,
                batch_size=2, sequence_length=3,
                method=CE.Method.FASTTEXT, grapharea_size=32):

    subfolder = F.FOLDER_SENSELABELED if slc_or_text_corpus else F.FOLDER_STANDARDTEXT
    graph_folder = os.path.join(F.FOLDER_GRAPH, subfolder)
    inputdata_folder = os.path.join(F.FOLDER_INPUT, subfolder)
    vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, subfolder)
    folders = (graph_folder, inputdata_folder, vocabulary_folder)

    # -------------------- 1: Setting up the graph, grapharea_matrix and vocabulary --------------------
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix = get_objects(slc_or_text_corpus,folders, method, grapharea_size)
    objects = graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix

    # -------------------- 2: Loading / creating the model --------------------
    torch.manual_seed(1) # for reproducibility while conducting experiments
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    if load_saved_model: # allows to load a model pre-trained on another dataset. Was not used for the paper results.
        model = load_model_from_file(slc_or_text_corpus, inputdata_folder, graph_dataobj)
    else:
        model = create_model(model_type, objects, include_globalnode_input, K, context_method, C, dim_qkv)

    # -------------------- 3: Moving objects on GPU --------------------
    logging.info("Graph-data object loaded, model initialized. Moving them to GPU device(s) if present.")

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        logging.info("Using " + str(n_gpu) + " GPUs")
        model = torch.nn.DataParallel(model, dim=0)
        model_forDataLoading = model.module
        batch_size = n_gpu if batch_size is None else batch_size  # if not specified, default batch_size = n. GPUs
    else:
        model_forDataLoading = model
        batch_size = 1 if batch_size is None else batch_size
    model.to(DEVICE)

    # -------------------- 4: Creating the DataLoaders for training, validation and test datasets --------------------
    corpus_fpath = os.path.join(F.FOLDER_MINICORPUSES, subfolder) # The only modification in a MiniExperiment
    datasets, dataloaders = get_dataloaders(objects, slc_or_text_corpus, corpus_fpath, folders,
                                  batch_size, sequence_length, model_forDataLoading)

    return model, datasets, dataloaders


################
def run_train(model,dataloaders, learning_rate, num_epochs, predict_senses=True, with_freezing=False):
    train_dataloader, _valid_dataloader, test_dataloader = dataloaders

    # -------------------- Setup; parameters and utilities --------------------
    Utils.init_logging('MiniExp-' + Utils.get_timestamp_month_to_sec() + '.log', loglevel=logging.INFO)
    slc_or_text = train_dataloader.dataset.sensecorpus_or_text

    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate)]

    model.train()

    polysense_thresholds = (2,3,5,10,30)
    polysense_globals_dict= AD.get_polysenseglobals_dict(slc_or_text, thresholds=polysense_thresholds)

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    parameters_ls = [param for param in model_forParameters.parameters()]
    model_forParameters.predict_senses = predict_senses
    logging.info(model)
    try:
        logging.info("Using K=" + str(model_forParameters.K))
        logging.info("C=" + str(model_forParameters.num_C))
        logging.info("context_method=" + str(model_forParameters.context_method))
    except Exception:
        pass # no further hyperparameters were specified

    steps_logging = 5
    overall_step = 0
    starting_time = time()

    freezing_epoch = (num_epochs // 3)*2 # we decide to freeze at 2/3rds of the number of epochs in the miniexperiment
    after_freezing_flag = False
    if with_freezing:
        model_forParameters.predict_senses = False
    weights_before_freezing_check_ls = []
    parameters_to_check_names_ls = []

    # debug
    torch.autograd.set_detect_anomaly(True)
    train_dataiter = iter(cycle(train_dataloader))
    test_dataiter = iter(cycle(test_dataloader))

    # -------------------- The training loop --------------------
    try: # to catch KeyboardInterrupt-s and still save the training & validation losses

        for epoch in range(1,num_epochs+1):

            optimizer = optimizers[-1] # pick the most recently created optimizer. Useful when freezing

            # -------------------- Initialization --------------------
            sum_epoch_loss_global = 0
            sum_epoch_loss_sense = 0
            sum_epoch_loss_polysense = 0

            epoch_step = 0
            epoch_senselabeled_tokens = 0
            epoch_polysense_tokens = 0

            predictions_history_dict = {'correct_g':0,
                                        'tot_g':0,
                                        'correct_all_s':0,
                                        'tot_all_s':0,
                                        'correct_poly_s': {}.fromkeys(polysense_thresholds,0),
                                        'tot_poly_s': {}.fromkeys(polysense_thresholds, 0)
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
                if epoch == 1:
                    log_input(batch_input, model_forParameters.last_idx_senses, slc_or_text)
                # starting operations on one batch
                optimizer.zero_grad()

                # compute loss for the batch
                (losses_tpl, num_sense_instances_tpl) = compute_model_loss(model, batch_input, batch_labels, predictions_history_dict,
                                                                           polysense_globals_dict, slc_or_text, verbose)
                loss_global, loss_sense, loss_polysense = losses_tpl
                num_batch_sense_tokens, num_batch_polysense_tokens = num_sense_instances_tpl

                # running sum of the training loss
                sum_epoch_loss_global = sum_epoch_loss_global + loss_global.item()
                if model_forParameters.predict_senses:
                    sum_epoch_loss_sense = sum_epoch_loss_sense + loss_sense.item() * num_batch_sense_tokens
                    epoch_senselabeled_tokens = epoch_senselabeled_tokens + num_batch_sense_tokens
                    sum_epoch_loss_polysense = sum_epoch_loss_polysense + loss_polysense.item() * num_batch_polysense_tokens
                    epoch_polysense_tokens = epoch_polysense_tokens + num_batch_polysense_tokens
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

                if with_freezing and epoch == freezing_epoch and not after_freezing_flag:
                    # we are predicting first the standard LM, and then the senses. Freeze (1), activate (2).
                    logging.info("New validation worse than previous one. " +
                                 "Freezing the weights in the standard LM, activating senses' prediction.")
                    logging.info("Status of parameters before freezing:")
                    for (name, p) in model_forParameters.named_parameters():  # (1)
                        logging.info("Parameter=" + name + " ; requires_grad=" + str(p.requires_grad))
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
                        logging.info("Parameter=" + name + " ; requires_grad=" + str(p.requires_grad))

                if verbose:
                    logging.info("Status of parameters:")
                    for (name, p) in model_forParameters.named_parameters():
                        logging.info("Parameter=" + name + " ; requires_grad=" + str(p.requires_grad))

                # end of an epoch.

            # -------------------- Computing training losses for the epoch--------------------
            logging.info("-----\n Training, end of epoch " + str(epoch) + ". Global step n." + str(overall_step) +
                         ". Time = " + str(round(time() - starting_time, 2)))
            logging.info("Training - Correct predictions / Total predictions:\n" + str(predictions_history_dict))

            epoch_sumlosses_tpl = sum_epoch_loss_global, sum_epoch_loss_sense, sum_epoch_loss_polysense
            epoch_numsteps_tpl = epoch_step, epoch_senselabeled_tokens, epoch_polysense_tokens
            Utils.record_statistics(epoch_sumlosses_tpl, epoch_numsteps_tpl)


    except KeyboardInterrupt:
        logging.info("Training loop interrupted manually by keyboard")

    # At the end: Evaluation on the test set
    evaluation(test_dataloader, test_dataiter, model, verbose=False, slc_or_text=slc_or_text)