import pandas as pd
import torch
import Filesystem as F
import Utils
from WordEmbeddings.ComputeEmbeddings import Method
import os
import numpy as np
import sqlite3
import logging

def load_senses_elements(embeddings_method, elements_name):
    senses_defs_fname = Utils.VECTORIZED + '_' + str(embeddings_method.value) + '_' + elements_name + '.npy'
    senses_defs_fpath = os.path.join(F.FOLDER_INPUT, senses_defs_fname)
    senses_defs_X = np.load(senses_defs_fpath)
    return senses_defs_X


def initialize_senses(X_defs, X_examples, average_or_random):

    db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()

    indicesTable_db_c.execute("SELECT * FROM vocabulary_table")
    current_row_idx = 0
    X_senses_ls = []
    logging.info("X_defs.shape=" + str(X_defs.shape))
    logging.info("X_examples.shape=" + str(X_examples.shape))

    while (True):
        db_row = indicesTable_db_c.fetchone()
        if db_row is None:
            break

        current_row_idx = current_row_idx + 1
        if average_or_random:
            start_defs = db_row[2]
            end_defs = db_row[3]
            defs_vectors = X_defs[start_defs:end_defs][:]
            start_examples = db_row[3]
            end_examples = db_row[4]
            examples_vectors = X_examples[start_examples:end_examples][:]
            all_vectors = np.concatenate([defs_vectors, examples_vectors])
            logging.debug("all_vectors.shape=" + str(all_vectors.shape) + "\n****")
            sense_vector = np.average(all_vectors, axis=0)
            X_senses_ls.extend([sense_vector])

    if average_or_random:
        X_senses = np.array(X_senses_ls)
    else: # if average_or_random == False:
        X_senses_random_ndarray = np.random.rand(current_row_idx, X_defs.shape[1]) # uniform distribution over [0, 1)
        X_senses_random = torch.from_numpy(X_senses_random_ndarray)
        X_senses = X_senses_random

    return X_senses





def exe():
    Utils.init_logging('temp.log')
    X_definitions = load_senses_elements(Method.FASTTEXT, Utils.DEFINITIONS)
    X_examples = load_senses_elements(Method.FASTTEXT, Utils.EXAMPLES)
    X_senses = initialize_senses(X_definitions, X_examples, average_or_random=True)
    return X_senses

