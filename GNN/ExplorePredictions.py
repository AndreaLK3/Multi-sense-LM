import os
import pandas as pd
import Filesystem as F
import Utils
import sqlite3
import torch
import logging

### The motivation for this module is the inability (at the time of writing) to achieve an epoch loss ==0
### when attempting to overfit on a very small training dataset. The best final epoch's loss was still ~1.2
###
### What tokens (globals and senses) am I predicting incorrectly?


### Given the numerical index of a global, return the corresponding word/token
def get_globalword_fromindex(global_index):
    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE)
    globals_vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')

    word = globals_vocabulary_df.iloc[global_index]['word']
    return word

### Given the numerical index of a sense, return the corresponding sense definition
def get_sense_fromindex(sense_index):
    senseindices_db = sqlite3.connect(os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB))
    senseindices_db_c = senseindices_db.cursor()

    senseindices_db_c.execute("SELECT word_sense FROM indices_table WHERE vocab_index="+ str(sense_index))
    sense_name_row = senseindices_db_c.fetchone()
    if sense_name_row is not None:
        sense_name = sense_name_row[0]
    else:
        sense_name = None
    return sense_name



def log_predictions(predictions_globals, predictions_senses, k=5):
    (values_g, indices_g) = predictions_globals.max(dim=0)
    (values_s, indices_s) = predictions_senses.max(dim=0)

    logging.info("The top- " + str(k) + " predicted globals are:")
    for i in range(k):
        word = get_globalword_fromindex(indices_g[i])
        score = values_g[i]
        logging.info("Word: " + word  +" ; score = " + str(round(score,5)))
    logging.info("\nThe top- " + str(k) + " predicted senses are:")
    for i in range(k):
        sense = get_sense_fromindex(values_s[i])
        score = values_s[i]
        logging.info("Word: " + sense + " ; score = " + str(round(score, 5)))

### Entry function, to invoke from RGCN
def log_solution_and_predictions(label_tpl, predictions_globals, predictions_senses, k):
    solution_global_idx = label_tpl[0]
    solution_sense_idx = label_tpl[1]

    nextglobal = get_globalword_fromindex(solution_global_idx)
    nextsense = get_sense_fromindex(solution_sense_idx)
    logging.info("Label: the next global is: " + str(nextglobal))
    logging.info("Label: the next sense is: " + str(nextsense))

    log_predictions(predictions_globals, predictions_senses, k)