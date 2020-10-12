import pandas as pd
import torch
import Filesystem as F
import Utils
import Graph.DefineGraphEdges as DGE
from VocabularyAndEmbeddings.ComputeEmbeddings import Method
import os
import numpy as np
import sqlite3
import logging
import torch_geometric
import re
from sklearn.decomposition import PCA


def load_senses_elements(elements_name, embeddings_method, use_PCA, inputdata_folder):
    if use_PCA:
        # the version reduced by PCA
        senses_elems_fpath = os.path.join(inputdata_folder, F.FOLDER_PCA,
                                          elements_name + '_' + str(embeddings_method.value) + '.npy')

    else:
        senses_elems_fname = Utils.VECTORIZED + '_' + str(embeddings_method.value) + '_' + elements_name + '.npy'
        senses_elems_fpath = os.path.join(inputdata_folder, senses_elems_fname)

    senses_elems_X = np.load(senses_elems_fpath)
    return torch.tensor(senses_elems_X).to(torch.float32)


# ------------ Auxiliary functions to initialize nodes ------------

def initialize_globals(E_embeddings, globals_vocabulary_ls, use_pca):

    if use_pca:
        pca = PCA(n_components=Utils.GRAPH_EMBEDDINGS_DIM)
        E_reduced_embeddings = pca.fit_transform(E_embeddings)
    else:
        E_reduced_embeddings = E_embeddings

    X_globals_ls = []

    for i in range(len(globals_vocabulary_ls)):
        # word = globals_vocabulary_ls[i]
        X_globals_ls.append(E_reduced_embeddings[i])

    X_globals = torch.stack(X_globals_ls, dim=0).to(torch.float32)
    return X_globals


def initialize_senses(X_defs, X_examples, X_globals, vocabulary_ls, average_or_random_flag, inputdata_folder):

    db_filepath = os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()

    indicesTable_db_c.execute("SELECT * FROM indices_table")
    current_row_idx = 0
    X_senses_ls = []
    num_dummy_senses = 0

    while True:
        db_row = indicesTable_db_c.fetchone()
        if db_row is None:
            break
        current_row_idx = current_row_idx + 1

        if average_or_random_flag:
            wn_id = db_row[0]

            pt = r'\.([^.])+\.(?=([0-9])+)'
            logging.debug("wn_id=" + str(wn_id))
            mtc = re.search(pt, db_row[0])
            pos = mtc.group(0)[1:-1]
            if pos == 'dummySense':
                # no definitions and examples, this is a dummy sense. It gets initialized with the global vector
                word = wn_id[0:Utils.get_locations_of_char(wn_id, '.')[-2]]
                global_idx = vocabulary_ls.index(word)
                sense_vector = X_globals[global_idx].numpy()
                X_senses_ls.extend([sense_vector])
                num_dummy_senses = num_dummy_senses+1

            else:
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

    if average_or_random_flag:
        X_senses = np.array(X_senses_ls)
    else:  # if average_or_random == False:
        X_senses_random_ndarray = np.random.rand(current_row_idx, X_defs.shape[1])  # uniform distribution over [0, 1)
        X_senses_random = torch.from_numpy(X_senses_random_ndarray)
        X_senses = X_senses_random

    return torch.tensor(X_senses), num_dummy_senses


# ----------------------------------------

def create_graph(method, slc_corpus):
    if method == Method.FASTTEXT:
        single_prototypes_file = F.SPVs_FASTTEXT_FILE
    elif method == Method.DISTILBERT:
        single_prototypes_file = F.SPVs_DISTILBERT_FILE
    else:
        logging.error("Method not implemented")
        raise AssertionError

    Utils.init_logging('DefineGraph_SLC'+ str(slc_corpus)+'.log')

    if slc_corpus:
        inputdata_folder = os.path.join(F.FOLDER_INPUT, F.FOLDER_SENSELABELED)
        vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, F.FOLDER_SENSELABELED)
        graph_folder = os.path.join(F.FOLDER_GRAPH, F.FOLDER_SENSELABELED)
    else:
        inputdata_folder = os.path.join(F.FOLDER_INPUT, F.FOLDER_STANDARDTEXT)
        vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, F.FOLDER_STANDARDTEXT)
        graph_folder = os.path.join(F.FOLDER_GRAPH, F.FOLDER_STANDARDTEXT)

    globals_vocabulary_fpath = os.path.join(vocabulary_folder, F.VOCABULARY_OF_GLOBALS_FILENAME)
    globals_vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')
    globals_vocabulary_ls = globals_vocabulary_df['word'].to_list().copy()

    E_embeddings = torch.tensor(np.load(os.path.join(inputdata_folder, single_prototypes_file))).to(torch.float32)
    logging.info("E_embeddings.shape=" + str(E_embeddings.shape))
    num_globals = E_embeddings.shape[0]
    embeddings_size = E_embeddings.shape[1]

    use_PCA = Utils.GRAPH_EMBEDDINGS_DIM < embeddings_size

    X_definitions = load_senses_elements(Utils.DEFINITIONS, method, use_PCA, inputdata_folder)
    X_examples = load_senses_elements(Utils.EXAMPLES, method, use_PCA, inputdata_folder)

    X_globals = initialize_globals(E_embeddings, globals_vocabulary_ls, use_PCA)
    X_senses, num_dummysenses = initialize_senses(X_definitions, X_examples, X_globals, globals_vocabulary_ls,
                                                  average_or_random_flag=True, inputdata_folder=inputdata_folder)
    num_senses = X_senses.shape[0]

    logging.info("X_senses.shape=" + str(X_senses.shape))
    logging.info("X_definitions.shape=" + str(X_definitions.shape))
    logging.info("X_examples.shape=" + str(X_examples.shape))
    logging.info("X_globals.shape=" + str(X_globals.shape))

    # The order for the index of the nodes:
    # sense=[0,se) ; single prototype=[se,se+sp) ; definitions=[se+sp, se+sp+d) ; examples=[se+sp+d, e==num_nodes)
    X = torch.cat([X_senses, X_globals, X_definitions, X_examples])

    # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges].
    # We can operate with a list of S-D tuples, adding t().contiguous()
    logging.info("Defining the edges: def, exs")
    definition_edges = DGE.get_edges_elements(Utils.DEFINITIONS, num_senses + num_globals, inputdata_folder)
    logging.info("definition_edges.__len__()=" + str(definition_edges.__len__())) # definition_edges.__len__()=25986
    example_edges = DGE.get_edges_elements(Utils.EXAMPLES, num_senses + num_globals + X_definitions.shape[0], inputdata_folder)
    logging.info("example_edges.__len__()=" + str(example_edges.__len__())) # example_edges.__len__()=26003

    logging.info("Defining the edges: lemma")
    lemma_edges = DGE.get_edges_lemmatized(globals_vocabulary_df, globals_vocabulary_ls, last_sense_idx=num_senses)

    logging.info("Defining the edges: senseChildren")
    senseChildren_edges = []
    # If operating on a sense-labeled corpus, we need to connect globals & their senses that do not belong to them
    if slc_corpus:
        sc_external_edges = DGE.get_additional_edges_sensechildren_from_slc(globals_vocabulary_df,
                                                                        globals_start_index_toadd=num_senses,
                                                                        inputdata_folder=inputdata_folder)
        senseChildren_edges.extend(sc_external_edges)
        logging.info("sc_edges_external_from_SLC.__len__()=" + str(sc_external_edges.__len__()))
    senseChildren_edges.extend(DGE.get_edges_sensechildren(globals_vocabulary_df, X_senses.shape[0], inputdata_folder))
    logging.info("sc_edges.__len__()=" + str(senseChildren_edges.__len__()))  # senseChildren_edges.__len__()=25986
    # We add self-loops to all globals that are not connected to any node.
    edges_selfloops = DGE.get_edges_selfloops(senseChildren_edges, lemma_edges, num_globals, num_senses)
    senseChildren_edges.extend(edges_selfloops)
    logging.info("sc_edges_with_selfloops.__len__()=" + str(senseChildren_edges.__len__()))

    logging.info("Defining the edges: syn, ant")
    syn_edges = DGE.get_edges_nyms(Utils.SYNONYMS, globals_vocabulary_df, num_senses, inputdata_folder)
    logging.info("syn_edges.__len__()=" + str(syn_edges.__len__()))
    ant_edges = DGE.get_edges_nyms(Utils.ANTONYMS, globals_vocabulary_df, num_senses, inputdata_folder)
    logging.info("ant_edges.__len__()=" + str(ant_edges.__len__()))

    edges_lts = torch.tensor(definition_edges + example_edges + senseChildren_edges + syn_edges + ant_edges + lemma_edges)
    edge_types = torch.tensor([0] * len(definition_edges) + [1] * len(example_edges) + [2] * len(senseChildren_edges) +
                              [3] * len(syn_edges) + [4] * len(ant_edges) + [5] * len(lemma_edges))
    node_types = torch.tensor([0] * X_senses.shape[0] + [1] * num_globals +
                              [2] * X_definitions.shape[0] + [3] * X_examples.shape[0])
    all_edges = edges_lts.t().contiguous()

    graph = torch_geometric.data.Data(x=X,
                                      edge_index=all_edges,
                                      edge_type=edge_types,
                                      node_types=node_types,
                                      num_relations=6)

    torch.save(graph, os.path.join(graph_folder, F.KBGRAPH_FILE))

    return graph



# Entry point function: try to load the graph, else create it if it does not exist
def get_graph_dataobject(new, slc_corpus, method=Method.FASTTEXT):
    if slc_corpus:
        graph_fpath = os.path.join(F.FOLDER_GRAPH, F.FOLDER_SENSELABELED, F.KBGRAPH_FILE)
    else:
        graph_fpath = os.path.join(F.FOLDER_GRAPH, F.FOLDER_STANDARDTEXT, F.KBGRAPH_FILE)
    if os.path.exists(graph_fpath) and not new:
        return torch.load(graph_fpath)
    else:
        graph_dataobj = create_graph(method, slc_corpus)
        torch.save(graph_dataobj, graph_fpath)
        return graph_dataobj