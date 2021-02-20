import Graph.Adjacencies as AD
import Graph.DefineGraph as DG
import NN.DataLoading as DL
import Filesystem as F
import os
import sqlite3
import Utils
import pandas as pd
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import SenseLabeledCorpus as SLC
import NN.NumericalIndices as NI
import numpy as np
from collections import deque
import logging
import warnings

def get_archives(subfolder):

    slc_or_text_corpus = True

    graph_folder = os.path.join(F.FOLDER_GRAPH, subfolder)
    inputdata_folder = os.path.join(F.FOLDER_INPUT, subfolder)
    vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, subfolder)

    globals_vocabulary_fpath = os.path.join(vocabulary_folder, F.VOCABULARY_OF_GLOBALS_FILENAME)
    vocab_df = pd.read_hdf(globals_vocabulary_fpath)

    senseindices_db_filepath = os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()

    graph_dataobj = DG.get_graph_dataobject(new=False, method=CE.Method.FASTTEXT, slc_corpus=slc_or_text_corpus)
    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, area_size=32, hops_in_area=1, graph_folder=graph_folder)
    E = DG.load_word_embeddings(inputdata_folder)

    return vocab_df, senseindices_db_c, grapharea_matrix, E



def compute_sense_ctx_averages(num_prev_words):
    # --------- Log and reset ---------

    slc_or_text_corpus = True
    subfolder = F.FOLDER_SENSELABELED if slc_or_text_corpus else F.FOLDER_STANDARDTEXT
    inputdata_folder = os.path.join(F.FOLDER_INPUT, subfolder)

    output_fname = str(num_prev_words) + F.MATRIX_SENSE_CONTEXTS_FILEEND
    output_filepath = os.path.join(inputdata_folder, output_fname)
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    # --------- Preparation for reading the SLC corpus ---------
    vocab_df, senseindices_db_c, grapharea_matrix, E = get_archives(subfolder)

    last_sense_idx = senseindices_db_c.execute("SELECT COUNT(*) from indices_table").fetchone()[0]
    first_idx_dummySenses = Utils.get_startpoint_dummySenses(slc_or_text_corpus)

    train_corpus_fpath = os.path.join(F.FOLDER_TEXT_CORPUSES, subfolder, F.FOLDER_TRAIN)

    # --------- Initializing the matrix that accumulates the context averages of sense occurrences, etc. ---------
    sense_ctx_avgs_A = np.zeros(shape=(last_sense_idx, E.shape[1]))
    occurrences_counter_vector = np.zeros(shape=last_sense_idx)

    # --------- Initialize queue for the current average of the preceding context ---------
    current_context_wordembs_q = deque()
    for i in range(num_prev_words-1):
        current_context_wordembs_q.append(np.zeros(shape=E.shape[1]))

    generator = SLC.read_split(train_corpus_fpath)
    next_token_tpl = None
    # --------- Reading the SLC corpus ---------
    while True:
        try:
            current_token_tpl, next_token_tpl = \
                NI.get_tokens_tpls(next_token_tpl, generator, senseindices_db_c, vocab_df, grapharea_matrix,
                                   last_sense_idx, first_idx_dummySenses, slc_or_text_corpus)
            global_idx, sense_idx = current_token_tpl

            global_embedding = E[global_idx]
            current_context_wordembs_q.append(global_embedding.numpy())

            context_stacked = np.stack(current_context_wordembs_q)
            current_context_avg = np.average(context_stacked, axis=0,
                               weights = [1 if np.count_nonzero(emb) > 0 else 0 for emb in current_context_wordembs_q ])
            sense_ctx_avgs_A[sense_idx] = sense_ctx_avgs_A[sense_idx] + current_context_avg
            occurrences_counter_vector[sense_idx] = occurrences_counter_vector[sense_idx] +1

            current_context_wordembs_q.popleft()

        except StopIteration:
            break

    for i in range(last_sense_idx):
        accumulated_ctx_averages = sense_ctx_avgs_A[i]
        num_occurrences = occurrences_counter_vector[i]

        if num_occurrences > 0:
            sense_ctx_average = accumulated_ctx_averages / num_occurrences
            sense_ctx_avgs_A[i] = sense_ctx_average

    np.save(output_filepath, sense_ctx_avgs_A)


