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
import NN.Models.SenseContext as SC
import NN.Models.SelfAttention as SA
from NN.Models.Common import ContextMethod

# ########## Auxiliary functions ##########

# ---------- a) Which variant of the model to use ----------
class ModelType(Enum):
    RNN = "RNN"
    SELECTK = "SelectK"
    SC = "Sense Context"
    SELFATT = "Self Attention Scores"

# ---------- b) The option to load a pre-trained version of one of our models ----------
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


# ########## Steps of setup_train ##########

# ---------- Step 1: setting up the graph, grapharea_matrix (used for speed) and the vocabulary  ----------
def get_objects(slc_or_text_corpus, gr_in_voc_folders, method=CE.Method.FASTTEXT, grapharea_size=32):
    graph_folder, inputdata_folder, vocabulary_folder = gr_in_voc_folders
    graph_dataobj = DG.get_graph_dataobject(new=False, method=method, slc_corpus=slc_or_text_corpus).to(DEVICE)

    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, grapharea_size, hops_in_area=1, graph_folder=graph_folder)

    single_prototypes_file = F.SPVs_FASTTEXT_FILE if method == Method.FASTTEXT else F.SPVs_DISTILBERT_FILE
    embeddings_matrix = torch.tensor(np.load(os.path.join(inputdata_folder, single_prototypes_file))).to(torch.float32)


    globals_vocabulary_fpath = os.path.join(vocabulary_folder, F.VOCABULARY_OF_GLOBALS_FILENAME)
    vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')

    vocabulary_numSensesList = vocabulary_df['num_senses'].to_list().copy()
    if all([num_senses == -1 for num_senses in vocabulary_numSensesList]):
        vocabulary_df = AD.compute_globals_numsenses(graph_dataobj, grapharea_matrix, grapharea_size,
                                                     slc_or_text_corpus)

    return graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix

# ---------- Step 2: Create the model, of the type we specify ----------
def create_model(model_type, objects, include_globalnode_input, K, context_method, C, dim_qkv):
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix = objects
    if model_type==ModelType.RNN:
        model = RNNs.RNN(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                           embeddings_matrix, include_globalnode_input,
                           batch_size=32, n_layers=3, n_hid_units=1024)
    elif model_type==ModelType.SELECTK:
        model = SelectK.SelectK(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                                include_globalnode_input, batch_size=32, n_layers=3, n_hid_units=1024, K=K)
    elif model_type==ModelType.SC:
        model = SC.SenseContext(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                                embeddings_matrix, include_globalnode_input, batch_size=32, n_layers=3,
                                n_hid_units=1024, K=K, num_C=C, context_method=context_method)
    elif model_type==ModelType.SELFATT:
        model = SA.ScoresLM(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
             include_globalnode_input, batch_size=32, n_layers=3, n_hid_units=1024, K=K,
                            num_C=C, context_method=context_method, dim_qkv=dim_qkv)
    else:
        raise Exception ("Model type specification incorrect")
    return model

# ---------- Step 4: Creating the dataloaders for training, validation and test sets ----------
# The high number of parameters allows to set up the dataloaders for experiments on another dataset
# (for instance on a small fragment during development)
def get_dataloaders(objects, slc_or_text_corpus, corpus_fpath, gr_in_voc_folders, batch_size, seq_len, model_forDataLoading):
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix = objects
    graph_folder, inputdata_folder, vocabulary_folder = gr_in_voc_folders

    senseindices_db_filepath = os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()

    bptt_collator = DL.BPTTBatchCollator(grapharea_size, seq_len)
    # globals_vocabulary_fpath = os.path.join(vocabulary_folder, F.VOCABULARY_OF_GLOBALS_FILENAME)
    # vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')
    train_corpus_fpath = os.path.join(corpus_fpath, F.FOLDER_TRAIN)
    valid_corpus_fpath = os.path.join(corpus_fpath, F.FOLDER_VALIDATION)
    test_corpus_fpath = os.path.join(corpus_fpath, F.FOLDER_TEST)
    train_dataset = DL.TextDataset(slc_or_text_corpus, train_corpus_fpath, senseindices_db_c, vocabulary_df,
                                   model_forDataLoading, grapharea_matrix, grapharea_size, graph_dataobj)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size * seq_len,
                                                   num_workers=0, collate_fn=bptt_collator)

    valid_dataset = DL.TextDataset(slc_or_text_corpus, valid_corpus_fpath, senseindices_db_c, vocabulary_df,
                                   model_forDataLoading, grapharea_matrix, grapharea_size, graph_dataobj)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size * seq_len,
                                                   num_workers=0, collate_fn=bptt_collator)

    test_dataset = DL.TextDataset(slc_or_text_corpus, test_corpus_fpath, senseindices_db_c, vocabulary_df,
                                  model_forDataLoading, grapharea_matrix, grapharea_size, graph_dataobj)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size * seq_len,
                                                   num_workers=0, collate_fn=bptt_collator)

    return (train_dataset, valid_dataset, test_dataset),(train_dataloader, valid_dataloader, test_dataloader)


################

def setup_train(slc_or_text_corpus, model_type, K=0, C=0, context_method=None,
                dim_qkv=300,
                include_globalnode_input=0, load_saved_model=False,
                batch_size=32, sequence_length=35,
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
    corpus_fpath = os.path.join(F.FOLDER_TEXT_CORPUSES, subfolder)
    datasets, dataloaders = get_dataloaders(objects, slc_or_text_corpus, corpus_fpath, folders,
                                  batch_size, sequence_length, model_forDataLoading)

    return model, datasets, dataloaders


################


################
def run_train(model, dataloaders, learning_rate, num_epochs, predict_senses=True, with_freezing=False):

    # -------------------- Step 1: Setup model --------------------
    Utils.init_logging('Training' + Utils.get_timestamp_month_to_sec() + '.log', loglevel=logging.INFO)
    train_dataloader, valid_dataloader, test_dataloader = dataloaders
    slc_or_text = train_dataloader.dataset.sensecorpus_or_text

    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate)]

    model.train()
    multisense_globals_set = set(AD.get_multisense_globals_indices(slc_or_text))

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    model_forParameters.predict_senses = predict_senses
    hyperparams_str = write_doc_logging(train_dataloader, model, model_forParameters, learning_rate)
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
    previous_valid_loss_senses = inf
    freezing_epoch = None
    after_freezing_flag = False
    if with_freezing:
        model_forParameters.predict_senses = False

    # torch.autograd.set_detect_anomaly(True) # for debug
    train_dataiter = iter(cycle(train_dataloader))
    valid_dataiter = iter(cycle(valid_dataloader))
    test_dataiter = iter(cycle(test_dataloader))

    # -------------------- Step 3) The training loop, for each epoch --------------------
    try: # to catch KeyboardInterrupt-s and still save the model

        for epoch in range(1,num_epochs+1):

            optimizer = optimizers[-1]  # pick the most recently created optimizer. Useful if using the freezing option

            # -------------------- Step 3a) Initialization --------------------
            sum_epoch_loss_global = 0
            sum_epoch_loss_senses = 0
            sum_epoch_loss_polysenses = 0

            epoch_step = 0
            epoch_senselabeled_tokens = 0
            epoch_polysense_tokens = 0

            correct_predictions_dict = {'correct_g':0,
                                        'tot_g':0,
                                        'correct_all_s':0,
                                        'tot_all_s':0,
                                        'correct_poly_s':0,
                                        'tot_poly_s':0
                                        }
            verbose = True if (epoch==num_epochs) or (epoch% 200==0) else False # deciding: log prediction output
            flag_earlystop = False

            # -------------------- Step 3b) Evaluation on the validation set --------------------
            if epoch>1:
                logging.info("After training " + str(epoch-1) + " epochs, validation:")
                valid_loss_globals, valid_loss_senses, polysenses_evaluation_loss = evaluation(valid_dataloader,
                                                                                                valid_dataiter,
                                                                                                model, slc_or_text,
                                                                                                verbose=False)

                # -------------- 3c) Check the validation loss & the need for freezing / early stopping --------------
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
                            optimizer, after_freezing_flag = freeze(optimizers, model, model_forParameters, learning_rate)
                            freezing_epoch = epoch
                if after_freezing_flag:
                    if (exp(valid_loss_senses) > exp(previous_valid_loss_senses) + 1) and epoch > freezing_epoch + 2:
                        # when properly implementing freezing, this should change into a check on the accuracy
                        logging.info("Early stopping on senses.")
                        flag_earlystop = True

                best_valid_loss_globals = min(best_valid_loss_globals, valid_loss_globals)
                previous_valid_loss_senses = valid_loss_senses
                best_valid_loss_senses = min(best_valid_loss_senses, valid_loss_senses)

                if epoch == num_epochs:
                    write_doc_logging(train_dataloader, model, model_forParameters, learning_rate)

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
                (losses_tpl, num_sense_instances_tpl), _ = compute_model_loss(model, batch_input, batch_labels, correct_predictions_dict,
                                                                           multisense_globals_set,slc_or_text, verbose)
                loss_global, loss_sense, loss_multisense = losses_tpl
                num_batch_sense_tokens, num_batch_multisense_tokens = num_sense_instances_tpl

                # running sum of the training loss
                sum_epoch_loss_global = sum_epoch_loss_global + loss_global.item()
                if model_forParameters.predict_senses:
                    sum_epoch_loss_senses = sum_epoch_loss_senses + loss_sense.item() * num_batch_sense_tokens
                    epoch_senselabeled_tokens = epoch_senselabeled_tokens + num_batch_sense_tokens
                    sum_epoch_loss_polysenses = sum_epoch_loss_polysenses + loss_multisense.item() * num_batch_multisense_tokens
                    epoch_polysense_tokens = epoch_polysense_tokens + num_batch_multisense_tokens
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

            # -------------------- 3e) Computing training losses for the epoch--------------------
            logging.info("-----\n Training, end of epoch " + str(epoch) + ". Global step n." + str(overall_step) +
                         ". Time = " + str(round(time() - starting_time, 2)))
            logging.info("Training - Correct predictions / Total predictions:\n" + str(correct_predictions_dict))

            epoch_sumlosses_tpl = sum_epoch_loss_global, sum_epoch_loss_senses, sum_epoch_loss_polysenses
            epoch_numsteps_tpl = epoch_step, epoch_senselabeled_tokens, epoch_polysense_tokens
            Utils.record_statistics(epoch_sumlosses_tpl, epoch_numsteps_tpl)


    except KeyboardInterrupt:
        logging.info("Training loop interrupted manually by keyboard")

    # --------------------- 4) Saving model --------------------
    torch.save(model, os.path.join(F.FOLDER_NN, hyperparams_str + '_end.model'))

    # --------------------- 5) At the end: Evaluation on the test set  ---------------------
    evaluation(test_dataloader, test_dataiter, model, verbose=False, slc_or_text=slc_or_text)
    logging.info("Final time=" + str(round(time() - starting_time, 2)))


# ##########
# Auxiliary function: Evaluation on a given dataset, e.g. validation or test set
def evaluation(evaluation_dataloader, evaluation_dataiter, model, slc_or_text, verbose):
    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__=="DataParallel" else model
    including_senses = model_forParameters.predict_senses
    polysemous_globals_set = set(AD.get_multisense_globals_indices(slc_or_text))

    model.eval()  # do not train the model now
    sum_eval_loss_globals = 0
    sum_eval_loss_senses = 0
    sum_eval_loss_polysenses = 0
    eval_correct_predictions_dict = {'correct_g':0,
                                    'tot_g':0,
                                    'correct_all_s':0,
                                    'tot_all_s':0,
                                    'correct_poly_s':0,
                                    'tot_poly_s':0
                                    }

    evaluation_step = 0
    evaluation_senselabeled_tokens = 0
    evaluation_polysense_tokens = 0
    logging_step = 500

    with torch.no_grad():  # Deactivates the autograd engine entirely to save some memory
        for b_idx in range(len(evaluation_dataloader)-1):
            batch_input, batch_labels = evaluation_dataiter.__next__()
            batch_input = batch_input.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            (losses_tpl, num_sense_instances_tpl), _ = compute_model_loss(model, batch_input, batch_labels, eval_correct_predictions_dict,
                                                                       polysemous_globals_set, slc_or_text, verbose=verbose)
            loss_globals, loss_senses, loss_polysenses = losses_tpl
            num_batch_sense_tokens, num_batch_polysense_tokens = num_sense_instances_tpl
            sum_eval_loss_globals = sum_eval_loss_globals + loss_globals.item()

            if including_senses:
                sum_eval_loss_senses = sum_eval_loss_senses + loss_senses.item() * num_batch_sense_tokens
                evaluation_senselabeled_tokens = evaluation_senselabeled_tokens + num_batch_sense_tokens
                sum_eval_loss_polysenses = sum_eval_loss_polysenses + loss_polysenses.item() * num_batch_polysense_tokens
                evaluation_polysense_tokens = evaluation_polysense_tokens + num_batch_polysense_tokens

            evaluation_step = evaluation_step + 1
            if evaluation_step % logging_step == 0:
                logging.info("Evaluation step n. " + str(evaluation_step))
                gc.collect()

    globals_evaluation_loss = sum_eval_loss_globals / evaluation_step
    if including_senses:
        senses_evaluation_loss = sum_eval_loss_senses / evaluation_senselabeled_tokens
        polysenses_evaluation_loss = sum_eval_loss_polysenses / evaluation_polysense_tokens
    else:
        senses_evaluation_loss = 0
        polysenses_evaluation_loss = 0

    logging.info("Evaluation - Correct predictions / Total predictions:\n" + str(eval_correct_predictions_dict))
    epoch_sumlosses_tpl = sum_eval_loss_globals, sum_eval_loss_senses, sum_eval_loss_polysenses
    epoch_numsteps_tpl = evaluation_step, evaluation_senselabeled_tokens, evaluation_polysense_tokens
    Utils.record_statistics(epoch_sumlosses_tpl, epoch_numsteps_tpl)

    model.train()  # training can resume

    return globals_evaluation_loss, senses_evaluation_loss, polysenses_evaluation_loss

# Auxiliary function: freezing the standard-LM part of a model, unfreezing the senses' part. Implementation incomplete
def freeze(optimizers, model, model_forParameters, learning_rate):
    # In this setting, we are optimizing first the standard LM, and then the senses.
    # Now: Freeze (1), activate (2).
    logging.info("New validation worse than previous one. " +
                 "Freezing the weights in the standard LM, activating senses' prediction.")

    weights_before_freezing_check_ls = []
    parameters_to_check_names_ls = []
    logging.info("Status of parameters before freezing:")
    for (name, p) in model_forParameters.named_parameters():  # (1)
        logging.info("Parameter=" + str(name) + " ; requires_grad=" + str(p.requires_grad))
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
        logging.info("Parameter=" + str(name) + " ; requires_grad=" + str(p.requires_grad))

    return optimizer, after_freezing_flag



