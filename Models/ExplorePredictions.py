import os
import pandas as pd

import Filesystem
import Filesystem as F
import Utils
import sqlite3
import torch
import logging
from math import exp

### The motivation for this module is the inability (at the time of writing) to achieve an epoch loss ==0
### when attempting to overfit on a very small training dataset. The best final epoch's loss was still ~1.2
###
### What tokens (globals and senses) am I predicting incorrectly?


### Given the numerical index of a global, return the corresponding word/token
def get_globalword_fromindex_df(global_index, vocabulary_folder):

    globals_vocabulary_fpath = os.path.join(vocabulary_folder, "vocabulary.h5")
    globals_vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')

    word = globals_vocabulary_df.iloc[global_index]['word']
    return word


### Given the numerical index of a sense, return the corresponding sense definition
def get_sense_fromindex(sense_index, inputdata_folder):

    senseindices_db = sqlite3.connect(os.path.join(inputdata_folder, Filesystem.INDICES_TABLE_DB))
    senseindices_db_c = senseindices_db.cursor()

    senseindices_db_c.execute("SELECT word_sense FROM indices_table WHERE vocab_index="+ str(sense_index))
    sense_name_row = senseindices_db_c.fetchone()
    if sense_name_row is not None:
        sense_name = sense_name_row[0]
    else:
        sense_name = None
    return sense_name


### Logging the predictions for : globals
def log_predicted_globals(predictions_globals, k, vocabulary_folder):
    (values_g, indices_g) = predictions_globals.sort(dim=0, descending=True)[0:k]

    logging.info("The top- " + str(k) + " predicted globals are:")
    for i in range(k):
        word = get_globalword_fromindex_df(indices_g[i].item(), vocabulary_folder)
        score = values_g[i].item()
        probability = round(exp(score) * 100,2)
        logging.info("Word: " + word  +" ; p=" + str(probability) + "%")

### Logging the predictions for : senses
def log_predicted_senses(predictions_senses, k, inputdata_folder):
    (values_s, indices_s) = predictions_senses.sort(dim=0, descending=True)[0:k]

    logging.info("The top- " + str(k) + " predicted senses are:")
    for i in range(k):
        sense = get_sense_fromindex(indices_s[i].item(), inputdata_folder)
        score = values_s[i].item()
        probability = round(exp(score) * 100, 2)
        if probability > 1:
            logging.info("Sense: " + sense + " ; p = " + str(probability) + "%")


### logs the solution and prediction for 1 sample
def log_solution_and_predictions(label_tpl, predictions_globals, predictions_senses, k, vocab_sources_ls, sp_method):
    solution_global_idx = label_tpl[0].item()
    solution_sense_idx = label_tpl[1].item()

    _, inputdata_folder, vocabulary_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)

    nextglobal = get_globalword_fromindex_df(solution_global_idx, vocabulary_folder)
    nextsense = get_sense_fromindex(solution_sense_idx, inputdata_folder)
    logging.info("\nLabel: the next global is: " + str(nextglobal) + "(from " + str(solution_global_idx) + ")")
    logging.info("Label: the next sense is: " + str(nextsense)  + "(from " + str(solution_sense_idx) + ")")

    log_predicted_globals(predictions_globals, k, vocabulary_folder)
    if nextsense is not None and len(predictions_senses.shape)!=0:
        log_predicted_senses(predictions_senses, k, inputdata_folder)


# Entry function, to invoke from RGCN. Batch level
def log_batch(labels_t, predictions_globals_t, predictions_senses_t, k, vocab_sources_ls, sp_method):
    batch_size = predictions_globals_t.shape[0]
    logging.info("log_batch:")
    logging.info("predictions_senses_t.shape=" + str(predictions_senses_t.shape))
    for i in range(batch_size):
        sample_labels = labels_t[i]
        sample_predglobals = predictions_globals_t[i]
        sample_predsenses = predictions_senses_t[i]
        log_solution_and_predictions(sample_labels, sample_predglobals, sample_predsenses, k, vocab_sources_ls, sp_method)
