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
import Vocabulary.Vocabulary_Utilities as VocabUtils
import SenseLabeledCorpus as SLC
from GNN.NumericalIndices import try_to_get_wordnet_sense
from PrepareKBInput.LemmatizeNyms import lemmatize_term
import nltk
import re


def load_senses_elements(embeddings_method, elements_name):
    senses_elems_fname = Utils.VECTORIZED + '_' + str(embeddings_method.value) + '_' + elements_name + '.npy'
    senses_elems_fpath = os.path.join(F.FOLDER_INPUT, senses_elems_fname)
    senses_elems_X = np.load(senses_elems_fpath)
    return torch.tensor(senses_elems_X).to(torch.float32)


# ------ Auxiliary functions to initialize nodes / create edges

def initialize_senses(X_defs, X_examples, X_globals, vocabulary_ls, average_or_random_flag):

    db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()

    indicesTable_db_c.execute("SELECT * FROM indices_table")
    current_row_idx = 0
    X_senses_ls = []

    while (True):
        db_row = indicesTable_db_c.fetchone()
        if db_row is None:
            break
        current_row_idx = current_row_idx + 1

        if average_or_random_flag:
            wn_id = db_row[0]

            pt = r'\.([^.])+\.'
            logging.info("wn_id=" + str(wn_id))
            mtc = re.search(pt, db_row[0])
            pos = mtc.group(0)[1:-1]
            if pos == 'Global':
                # no definitions and examples, this is a dummy sense. It gets initialized with the global vector
                word = wn_id[0:wn_id.find('.')] if wn_id[0] != '.' else '.'
                global_idx = vocabulary_ls.index(word)
                sense_vector = X_globals[global_idx].numpy()
                X_senses_ls.extend([sense_vector])

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
            logging.info(sense_vector.dtype)

    if average_or_random_flag:
        X_senses = np.array(X_senses_ls)
    else: # if average_or_random == False:
        X_senses_random_ndarray = np.random.rand(current_row_idx, X_defs.shape[1]) # uniform distribution over [0, 1)
        X_senses_random = torch.from_numpy(X_senses_random_ndarray)
        X_senses = X_senses_random

    return torch.tensor(X_senses)


# definitions -> senses : [se+sp, se+sp+d) -> [0,se)
# examples --> senses : [se+sp+d, e==num_nodes) -> [0,se)
def get_edges_elements(elements_name, elements_start_index_toadd):
    db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()

    indicesTable_db_c.execute("SELECT * FROM indices_table")
    edges_ls = []
    edges_toadd_counter = 0
    while (True):
        db_row = indicesTable_db_c.fetchone()
        if db_row is None:
            break
        target_idx = db_row[1]
        if elements_name==Utils.DEFINITIONS:
            start_sources = db_row[2] + elements_start_index_toadd
            end_sources = db_row[3] + elements_start_index_toadd
        else: # if elements_name==Utils.EXAMPLES:
            start_sources = db_row[4] + elements_start_index_toadd
            end_sources = db_row[5] + elements_start_index_toadd
        edges_toadd_counter = edges_toadd_counter + (end_sources-start_sources)
        for source in range(start_sources, end_sources):
            edges_ls.append((source, target_idx))

    indicesTable_db.close()
    return edges_ls


# global -> senses : [se,se+sp) -> [0,se)
def get_edges_sensechildren(globals_voc_df, globals_start_index_toadd):

    db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()
    indicesTable_db_c.execute("SELECT * FROM indices_table")

    edges_ls = []

    while (True):
        db_row = indicesTable_db_c.fetchone()
        if db_row is None:
            break

        word_sense = db_row[0]
        word = Utils.get_word_from_sense(word_sense)
        sourceglobal_raw_idx = globals_voc_df.loc[globals_voc_df['word'] == word].index[0]
        sourceglobal_idx = globals_start_index_toadd + sourceglobal_raw_idx
        targetsense_idx = db_row[1]

        edges_ls.append((sourceglobal_idx, targetsense_idx))

    indicesTable_db.close()
    return edges_ls


def get_additional_edges_sensechildren_from_slc(globals_voc_df, globals_start_index_toadd):
    logging.info("Reading the sense-labeled corpus, to create the connections between globals"
                 " and the senses that belong to other words.")

    slc_train_corpus_gen = SLC.read_split(Utils.TRAINING)
    senseindices_db = sqlite3.connect(os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB))
    senseindices_db_c = senseindices_db.cursor()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    edges_to_add_ls = []
    #words_and_senses_ls = [] # for debug purposes

    try:
        while True:
            token_dict = slc_train_corpus_gen.__next__()
            keys = token_dict.keys()
            sense_index_queryresult = None

            # 1) Get the sense (and its index) specified in the SLC for the current token
            if 'wn30_key' in keys:
                wn30_key = token_dict['wn30_key']
                wordnet_sense = try_to_get_wordnet_sense(wn30_key)
                if wordnet_sense is not None:
                    try:
                        query = "SELECT vocab_index FROM indices_table " + "WHERE word_sense='" + wordnet_sense + "'"
                        sense_index_queryresult = senseindices_db_c.execute(query).fetchone()
                    except sqlite3.OperationalError:
                        logging.info("Error while attempting to execute query: " + query + " . Skipping sense")

                if sense_index_queryresult is None:  # there was no sense-key, or we did not find the sense for the key
                    continue # do nothing, we do not add a global-sense connection
                else:
                    targetsense_idx = sense_index_queryresult[0]
            else:
                continue # there was no sense-key specified for this token
            # 2) Get the global word of this token
            word = VocabUtils.process_word_token(token_dict)  # html.unescape
            lemmatized_word = lemmatize_term(word, lemmatizer)# since currently we always lemmatize in SelectK and other sense architectures
            if lemmatized_word not in wordnet_sense: # we are connecting all the "external" senses, e.g. say->state.v.01
                try:
                    sourceglobal_absolute_idx = globals_voc_df.loc[globals_voc_df['word'] == lemmatized_word].index[0]
                    global_relative_X_index = globals_start_index_toadd + sourceglobal_absolute_idx
                    edges_to_add_ls.append((global_relative_X_index, targetsense_idx))
                    #words_and_senses_ls.append((lemmatized_word, wordnet_sense))# for debug purposes
                except IndexError:  # global not present. No need to redirect onto <unk>, we skip
                    pass
            # else, we do not connect again the internal senses, e.g. say->say.v.01, we did that already
    except StopIteration:
        pass
    # remove duplicates
    edges_to_add_ls = list(set(edges_to_add_ls))
    #words_and_senses_ls = list(set(words_and_senses_ls))
    return edges_to_add_ls


# Since GATs and other Graph Neural Networks do not allow for nodes without edges, we add self-loops to all
# stopwords like for, of, etc. They are ignored by the message-passing framework
def get_edges_selfloops(sc_edges, num_globals):
    globals_sources = sorted(list(map(lambda edge_tpl : edge_tpl[0], sc_edges)))
    senses_targets = list(map(lambda edge_tpl : edge_tpl[1], sc_edges))
    max_sense = max(senses_targets)

    all_globals_indices = list(range(max_sense+1,max_sense+num_globals+1))

    globals_needing_selfloop = [g_idx for g_idx in all_globals_indices if g_idx not in globals_sources]

    edges_selfloops = [(g_idx, g_idx) for g_idx in globals_needing_selfloop]

    return edges_selfloops


# Synonyms and antonyms: global -> global : [se,se+sp) -> [se,se+sp).
# Bidirectional (which means 2 connections, (a,b) and (b,a)
def get_edges_nyms(nyms_name, globals_voc_df, globals_start_index_toadd):
    nyms_archive_fname = Utils.PROCESSED + '_' + nyms_name + '.h5'
    nyms_archive_fpath = os.path.join(F.FOLDER_INPUT, nyms_archive_fname)

    nyms_df = pd.read_hdf(nyms_archive_fpath, key=nyms_name, mode="r")
    edges_ls = []

    counter = 0
    for tpl in nyms_df.itertuples():
        word_sense = tpl.sense_wn_id
        word1 = Utils.get_word_from_sense(word_sense)
        word1 = VocabUtils.process_word_token({'surface_form': word1})
        try:
            global_raw_idx_1 = globals_voc_df.loc[globals_voc_df['word'] == word1].index[0]
        except IndexError:
            logging.debug("Edges>" + nyms_name + ". Word '" + word1 + "' not found in globals' vocabulary. Skipping...")
            continue
        global_idx_1 = globals_start_index_toadd + global_raw_idx_1

        word2 = getattr(tpl, nyms_name)
        word2 = VocabUtils.process_word_token({'surface_form': word2})
        try:
            global_raw_idx_2 = globals_voc_df.loc[globals_voc_df['word'] == word2].index[0]
        except IndexError:
            logging.debug("Edges>" + nyms_name + ". Word '" + word2 + "' not found in globals' vocabulary. Skipping...")
            continue
        global_idx_2 = globals_start_index_toadd + global_raw_idx_2

        edges_ls.append((global_idx_1, global_idx_2))
        edges_ls.append((global_idx_2, global_idx_1))
        counter = counter + 1
        if counter % 5000 == 0:
            logging.info("Inserted " + str(counter) + " "  + nyms_name + " edges")

    return edges_ls

# ----------------------------------------

def create_graph(method, slc_corpus):
    if method == Method.FASTTEXT:
        single_prototypes_file = F.SPVs_FASTTEXT_FILE
    elif method == Method.DISTILBERT:
        single_prototypes_file = F.SPVs_DISTILBERT_FILE
    else:
        logging.error("Method not implemented")
        raise AssertionError

    Utils.init_logging('DefineGraph.log')

    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE )
    globals_vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')
    globals_vocabulary_ls = globals_vocabulary_df['word'].to_list().copy()

    X_definitions = load_senses_elements(method, Utils.DEFINITIONS)
    X_examples = load_senses_elements(method, Utils.EXAMPLES)
    X_globals = torch.tensor(np.load(os.path.join(F.FOLDER_INPUT, single_prototypes_file))).to(torch.float32)
    X_senses = initialize_senses(X_definitions, X_examples, X_globals, globals_vocabulary_ls, average_or_random_flag=True)

    logging.info("Constructing X, matrix of node features")
    logging.info("X_definitions.shape=" + str(X_definitions.shape)) # X_definitions.shape=torch.Size([25986, 300])
    logging.info("X_examples.shape=" + str(X_examples.shape)) # X_examples.shape=torch.Size([26003, 300])
    logging.info("X_senses.shape=" + str(X_senses.shape)) # X_senses.shape=torch.Size([25986, 300])
    logging.info("X_globals.shape=" + str(X_globals.shape)) # X_globals.shape=torch.Size([21988, 300])

    # The order for the index of the nodes:
    # sense=[0,se) ; single prototype=[se,se+sp) ; definitions=[se+sp, se+sp+d) ; examples=[se+sp+d, e==num_nodes)
    X = torch.cat([X_senses, X_globals, X_definitions, X_examples])
    # Currently, total number of nodes: 71882

    # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges].
    # We can operate with a list of S-D tuples, adding t().contiguous()
    logging.info("Defining the edges: def, exs")
    def_edges_se = get_edges_elements(Utils.DEFINITIONS, X_senses.shape[0] + X_globals.shape[0])
    logging.info("def_edges_se.__len__()=" + str(def_edges_se.__len__())) # def_edges_se.__len__()=25986
    exs_edges_se = get_edges_elements(Utils.EXAMPLES, X_senses.shape[0] + X_globals.shape[0]+ X_definitions.shape[0])
    logging.info("exs_edges_se.__len__()=" + str(exs_edges_se.__len__())) # exs_edges_se.__len__()=26003

    logging.info("Defining the edges: sc")
    sc_edges = get_edges_sensechildren(globals_vocabulary_df, X_senses.shape[0])
    logging.info("sc_edges.__len__()=" + str(sc_edges.__len__())) # sc_edges.__len__()=25986
    # If operating on a sense-labeled corpus, we need to connect globals & their senses that do not belong to them
    if slc_corpus:
        sc_external_edges = get_additional_edges_sensechildren_from_slc(globals_vocabulary_df, globals_start_index_toadd=X_senses.shape[0])
        sc_edges.extend(sc_external_edges)
        logging.info("sc_edges_with_external.__len__()=" + str(sc_external_edges.__len__()))
    # We add self-loops to all globals without a sense.
    edges_selfloops = get_edges_selfloops(sc_edges, num_globals=X_globals.shape[0])
    sc_edges.extend(edges_selfloops)
    logging.info("sc_edges_with_selfloops.__len__()=" + str(sc_edges.__len__()))

    logging.info("Defining the edges: syn, ant")
    syn_edges = get_edges_nyms(Utils.SYNONYMS, globals_vocabulary_df, X_senses.shape[0])
    logging.info("syn_edges.__len__()=" + str(syn_edges.__len__()))
    ant_edges = get_edges_nyms(Utils.ANTONYMS, globals_vocabulary_df, X_senses.shape[0])
    logging.info("ant_edges.__len__()=" + str(ant_edges.__len__()))

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

    torch.save(graph, os.path.join('GNN', F.KBGRAPH_FILE))

    return graph



# Entry point function: try to load the graph, else create it if it does not exist
def get_graph_dataobject(new=False, method=Method.FASTTEXT, slc_corpus=False):
    graph_fpath = os.path.join('GNN', F.KBGRAPH_FILE)
    if os.path.exists(graph_fpath) and not new:
        return torch.load(graph_fpath)
    else:
        graph_dataobj = create_graph(method, slc_corpus)
        torch.save(graph_dataobj, graph_fpath)
        return graph_dataobj