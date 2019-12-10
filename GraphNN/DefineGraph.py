import pandas as pd
import torch
import Filesystem as F
import Utils
from WordEmbeddings.ComputeEmbeddings import Method
import os
import numpy as np
import sqlite3
import logging
import torch_geometric

def load_senses_elements(embeddings_method, elements_name):
    senses_defs_fname = Utils.VECTORIZED + '_' + str(embeddings_method.value) + '_' + elements_name + '.npy'
    senses_defs_fpath = os.path.join(F.FOLDER_INPUT, senses_defs_fname)
    senses_defs_X = np.load(senses_defs_fpath)
    return senses_defs_X


def initialize_senses(X_defs, X_examples, average_or_random):

    db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()

    indicesTable_db_c.execute("SELECT * FROM indices_table")
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


# definitions -> senses : [se+sp, se+sp+d) -> [0,se)
# examples --> senses : [se+sp+d, e==num_nodes) -> [0,se)
def get_edges_elements(elements_name, elements_start_index_toadd):
    db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()

    indicesTable_db_c.execute("SELECT * FROM indices_table")
    edges_ls = []

    while (True):
        db_row = indicesTable_db_c.fetchone()
        if db_row is None:
            break

        target_idx = db_row[1]
        if elements_name==Utils.DEFINITIONS:
            start_sources = db_row[2] + elements_start_index_toadd
            end_sources = db_row[3] + elements_start_index_toadd
        else: # if elements_name==Utils.EXAMPLES:
            start_sources = db_row[3] + elements_start_index_toadd
            end_sources = db_row[4] + elements_start_index_toadd
        for source in range(start_sources, end_sources):
            edges_ls.append((source, target_idx))

    indicesTable_db.close()
    return edges_ls


 # global -> senses : [se,se+sp) -> [0,se)
def get_edges_sensechildren(globals_vocabulary_fpath, globals_start_index_toadd):

    db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()
    indicesTable_db_c.execute("SELECT * FROM indices_table")

    globals_vocabulary_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')

    edges_ls = []

    while (True):
        db_row = indicesTable_db_c.fetchone()
        if db_row is None:
            break

        word_sense = db_row[0]
        word = Utils.get_word_from_sense(word_sense)
        sourceglobal_raw_idx = Utils.select_from_hdf5(globals_vocabulary_h5, 'vocabulary', ['word'], [word]).index[0]
        sourceglobal_idx = globals_start_index_toadd + sourceglobal_raw_idx
        targetsense_idx = db_row[1]

        edges_ls.append((sourceglobal_idx, targetsense_idx))

    indicesTable_db.close()
    return edges_ls


# Synonyms and antonyms: global -> global : [se,se+sp) -> [se,se+sp).
# Bidirectional (which means 2 connections, (a,b) and (b,a)
def get_edges_nyms(nyms_name, globals_vocabulary_fpath, globals_start_index_toadd):
    nyms_archive_fname = Utils.PROCESSED + '_' + nyms_name + '.h5'
    nyms_archive_fpath = os.path.join(F.FOLDER_INPUT, nyms_archive_fname)

    nyms_df = pd.read_hdf(nyms_archive_fpath, key=nyms_name, mode="r")
    globals_vocabulary_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')
    edges_ls = []

    for tpl in nyms_df.itertuples():
        word_sense = tpl.sense_wn_id
        word1 = Utils.get_word_from_sense(word_sense)
        global_raw_idx_1 = Utils.select_from_hdf5(globals_vocabulary_h5, 'vocabulary', ['word'], [word1]).index[0]
        global_idx_1 = globals_start_index_toadd + global_raw_idx_1

        word2 = getattr(tpl, nyms_name)
        global_raw_idx_2 = Utils.select_from_hdf5(globals_vocabulary_h5, 'vocabulary', ['word'], [word2]).index[0]
        global_idx_2 = globals_start_index_toadd + global_raw_idx_2

        edges_ls.append((global_idx_1, global_idx_2))
        edges_ls.append((global_idx_2, global_idx_1))

    globals_vocabulary_h5.close()
    return edges_ls


def create_graph():
    Utils.init_logging('temp.log')
    X_definitions = load_senses_elements(Method.FASTTEXT, Utils.DEFINITIONS)
    X_examples = load_senses_elements(Method.FASTTEXT, Utils.EXAMPLES)
    X_senses = initialize_senses(X_definitions, X_examples, average_or_random=True)
    X_globals = np.load(os.path.join(F.FOLDER_INPUT, F.SPVs_FASTTEXT_FILE))

    logging.info("Constructing X, matrix of node features")
    logging.info(X_definitions.shape)
    logging.info(X_examples.shape)
    logging.info(X_senses.shape)
    logging.info(X_globals.shape)

    # The order for the index of the nodes:
    # sense=[0,se) ; single prototype=[se,se+sp) ; definitions=[se+sp, se+sp+d) ; examples=[se+sp+d, e==num_nodes)
    X = np.concatenate([X_senses, X_globals, X_definitions, X_examples])

    # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges].
    # We can operate with a list of S-D tuples, adding t().contiguous()
    logging.info("Defining the edges: def, exs")
    def_edges_se = get_edges_elements(Utils.DEFINITIONS, X_senses.shape[0] + X_globals.shape[0])
    exs_edges_se = get_edges_elements(Utils.EXAMPLES, X_senses.shape[0] + X_globals.shape[0]+ X_definitions.shape[0])
    logging.info("Defining the edges: sc")
    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCAB_FROMSLC_FILE)
    sc_edges = get_edges_sensechildren(globals_vocabulary_fpath, X_senses.shape[0])
    logging.info("Defining the edges: syn, ant")
    syn_edges = get_edges_nyms(Utils.SYNONYMS, globals_vocabulary_fpath, X_senses.shape[0])
    ant_edges = get_edges_nyms(Utils.ANTONYMS, globals_vocabulary_fpath, X_senses.shape[0])

    edges_lts = torch.tensor(def_edges_se + exs_edges_se + sc_edges + syn_edges + ant_edges)
    edge_types = torch.tensor([0] * len(def_edges_se) + [1] * len(exs_edges_se) + [2] * len(sc_edges) +
                              [3] * len(syn_edges) + [4] * len(ant_edges))
    node_types = torch.tensor([0] * X_senses.shape[0] + [1] * X_globals.shape[0] +
                              [2] * X_definitions.shape[0] + [3] * X_examples.shape[0])
    all_edges = edges_lts.t().contiguous()

    graph = torch_geometric.data.Data(x=X,
                                      edge_index=all_edges,
                                      edge_type=edge_types,
                                      node_types=node_types,
                                      num_relations=5)

    torch.save(graph, F.KBGRAPH_FILE)

    return graph



# Entry point function: try to load the graph, else create it if it does not exist
def get_graph_dataobject():
    if os.path.exists(F.KBGRAPH_FILE):
        return torch.load(F.KBGRAPH_FILE)
    else:
        return create_graph()