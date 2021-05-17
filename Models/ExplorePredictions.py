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

    senseindices_db = sqlite3.connect(os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB))
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
    predicted_word_idx = indices_g[0].item()
    predicted_word = get_globalword_fromindex_df(predicted_word_idx, vocabulary_folder)

    log_string = "The top- " + str(k) + " predicted globals are:"
    predictions_log_ls = []
    for i in range(k):
        word = get_globalword_fromindex_df(indices_g[i].item(), vocabulary_folder)
        score = values_g[i].item()
        probability = round(exp(score) * 100,2)
        predictions_log_ls.append("Word: " + word  +", p=" + str(probability) + "%")
    log_string = log_string + " ; ".join(predictions_log_ls)
    logging.info(log_string)

    return predicted_word_idx, predicted_word

### Logging the predictions for : senses
def log_predicted_senses(predictions_senses, k, inputdata_folder):
    (values_s, indices_s) = predictions_senses.sort(dim=0, descending=True)[0:k]
    predicted_sense_idx = indices_s[0].item()
    predicted_sense = get_sense_fromindex(predicted_sense_idx, inputdata_folder)

    log_string = "The top- " + str(k) + " predicted senses are: "
    predictions_log_ls = []
    for i in range(k):
        sense = get_sense_fromindex(indices_s[i].item(), inputdata_folder)
        score = values_s[i].item()
        probability = round(exp(score) * 100, 2)
        if probability > 1:
            predictions_log_ls.append(sense +", p=" + str(probability) + "%")
    log_string = log_string + " ; ".join(predictions_log_ls)
    logging.info(log_string)

    return predicted_sense_idx, predicted_sense


### logs the solution and prediction for 1 sample
def log_solution_and_predictions(label_tpl, predictions_globals, predictions_senses, vocab_sources_ls, k=3):
    solution_global_idx = label_tpl[0].item()
    solution_sense_idx = label_tpl[1].item()

    _, inputdata_folder, vocabulary_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls)

    nextglobal = get_globalword_fromindex_df(solution_global_idx, vocabulary_folder)
    nextsense = get_sense_fromindex(solution_sense_idx, inputdata_folder)
    logging.info("\nLabels, next global =" + str(nextglobal) + "\t; next sense =" +  str(nextsense))

    predicted_word_idx, predicted_word = log_predicted_globals(predictions_globals, k, vocabulary_folder)
    correct_word = predicted_word if (predicted_word_idx == solution_global_idx) else None

    if nextsense is not None and len(predictions_senses.shape)!=0:
        predicted_sense_idx, predicted_sense = log_predicted_senses(predictions_senses, k, inputdata_folder)
        correct_sense = predicted_sense if (predicted_sense_idx == solution_sense_idx) else None
    else:
        correct_sense = None
    return correct_word, correct_sense


# Entry function, to invoke from RGCN. Batch level
def log_batch(labels_t, predictions_globals_t, predictions_senses_t, vocab_sources_ls):
    correct_words_ls = []
    correct_senses_ls = []
    batch_size = predictions_globals_t.shape[0]

    for i in range(batch_size):
        sample_labels = labels_t[i]
        sample_predglobals = predictions_globals_t[i]  # the probability distribution over the globals
        sample_predsenses = predictions_senses_t[i]  # the probability distribution over the senses
        word, sense = log_solution_and_predictions(sample_labels, sample_predglobals, sample_predsenses, vocab_sources_ls)
        correct_words_ls.append(word)
        correct_senses_ls.append(sense)

    correct_words_dict = {w: correct_words_ls.count(w) for w in correct_words_ls if w is not None}
    correct_senses_dict = {s: correct_senses_ls.count(s) for s in correct_senses_ls if s is not None}
    logging.info("correct_words_dict=" + str(correct_words_dict))
    logging.info("correct_senses_dict=" + str(correct_senses_dict))

    return correct_words_dict, correct_senses_dict
