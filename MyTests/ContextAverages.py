from NN.ContextAverages import get_archives
import Utils
import Filesystem as F
import os
import sqlite3
import Utils
import pandas as pd
import logging
import SenseLabeledCorpus as SLC
import NN.NumericalIndices as NI
import numpy as np
from collections import deque

def compute_sense_ctx_averages(num_prev_words):
    # --------- Log and reset ---------
    Utils.init_logging('ContextAverages.log')
    slc_or_text_corpus = True
    subfolder = F.FOLDER_SENSELABELED if slc_or_text_corpus else F.FOLDER_STANDARDTEXT

    output_fname = str(num_prev_words) + F.MATRIX_SENSE_CONTEXTS_FILEEND
    output_filepath = os.path.join(F.FOLDER_TEST, output_fname)
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    # --------- Preparation for reading the SLC corpus ---------
    vocab_h5, senseindices_db_c, grapharea_matrix, graph_dataobj = get_archives(subfolder)

    last_sense_idx = senseindices_db_c.execute("SELECT COUNT(*) from indices_table").fetchone()[0]
    first_idx_dummySenses = Utils.get_startpoint_dummySenses(slc_or_text_corpus)

    train_corpus_fpath = os.path.join(F.FOLDER_MINICORPUSES, subfolder, F.FOLDER_TRAIN)

    # --------- Initializing the matrix that accumulates the context averages of sense occurrences, etc. ---------
    sense_ctx_avgs_A = np.zeros(shape=(last_sense_idx, graph_dataobj.x.shape[1]))
    occurrences_counter_vector = np.zeros(shape=last_sense_idx)

    # --------- Initialize queue for the current average of the preceding context ---------
    current_context_wordembs_q = deque()
    for i in range(num_prev_words-1):
        current_context_wordembs_q.append(np.zeros(shape=graph_dataobj.x.shape[1]))

    generator = SLC.read_split(train_corpus_fpath)
    next_token_tpl = None
    # --------- Reading the SLC corpus ---------
    while True:
        try:
            current_token_tpl, next_token_tpl = \
                NI.get_tokens_tpls(next_token_tpl, generator, senseindices_db_c, vocab_h5, grapharea_matrix,
                                   last_sense_idx, first_idx_dummySenses, slc_or_text_corpus)
            global_idx, sense_idx = current_token_tpl
            inmatrix_global_idx = last_sense_idx + global_idx

            global_embedding = graph_dataobj.x[inmatrix_global_idx]
            current_context_wordembs_q.append(global_embedding.numpy())

            context_stacked = np.stack(current_context_wordembs_q)
            current_context_avg = np.average(context_stacked, axis=0,
                               weights = [1 if np.count_nonzero(emb) > 0 else 0 for emb in current_context_wordembs_q ])
            logging.info("***\nsense_idx=" + str(sense_idx) + " ; context_stacked[:,0:3]=" + str(context_stacked[:,0:3]) +
                         "\ncurrent_context_avg=" + str(current_context_avg[0:3]))
            sense_ctx_avgs_A[sense_idx] = sense_ctx_avgs_A[sense_idx] + current_context_avg
            occurrences_counter_vector[sense_idx] = occurrences_counter_vector[sense_idx] + 1

            current_context_wordembs_q.popleft()

        except StopIteration:
            break

    for i in range(last_sense_idx):
        accumulated_ctx_averages = sense_ctx_avgs_A[i]
        num_occurrences = occurrences_counter_vector[i]

        if num_occurrences > 0:
            sense_ctx_average = accumulated_ctx_averages / num_occurrences
            sense_ctx_avgs_A[i] = sense_ctx_average
            logging.info("sense_idx=i =" + str(i) + " ; num_occurrences=" + str(num_occurrences) +
                         " ; sense_ctx_avgs_A[i,0:3] =" + str(sense_ctx_avgs_A[i,0:3]))

    np.save(output_filepath, sense_ctx_avgs_A)