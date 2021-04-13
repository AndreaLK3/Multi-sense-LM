import Graph.DefineGraph as DG
import Graph.Adjacencies as AD
import Models.DataLoading.NumericalIndices as NI
import Utils
import logging
import os
import Filesystem as F
import pandas as pd
from Models.ExplorePredictions import get_sense_fromindex, get_globalword_fromindex_df
import sqlite3

def log_node(n, last_indices_tpl, inputdata_folder, vocabulary_folder, definitions_df, examples_df):
    last_idx_senses , last_idx_globals, last_idx_definitions = last_indices_tpl
    if n < last_idx_senses:
        sense = get_sense_fromindex(n, inputdata_folder)
        logging.info("n=" + str(n) + " ; sense=" + str(sense))
    elif last_idx_senses < n and n < last_idx_globals:
        idx = n - last_idx_senses
        global_word = get_globalword_fromindex_df(idx, vocabulary_folder)
        logging.info("n=" + str(n) + " ; global=" + str(global_word))
    elif last_idx_globals < n and n < last_idx_definitions:
        idx = (n - last_idx_globals)
        logging.info("definition: n=" + str(n) + str((definitions_df.iloc[idx].sense_wn_id, definitions_df.iloc[idx].definitions)))
    elif last_idx_definitions < n:
        idx = (n - last_idx_definitions)
        logging.info("example: n=" + str(n) + str((examples_df.iloc[idx].sense_wn_id, examples_df.iloc[idx].examples)))


def initialize_archives(vocab_sources_ls, sp_method):
    graph_folder, inputdata_folder, vocabulary_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)

    definitions_h5 = os.path.join(inputdata_folder, Utils.PROCESSED + '_' + Utils.DEFINITIONS + ".h5")
    definitions_df = pd.read_hdf(definitions_h5, key=Utils.DEFINITIONS, mode="r")
    examples_h5 = os.path.join(inputdata_folder, Utils.PROCESSED + '_' + Utils.EXAMPLES + ".h5")
    examples_df = pd.read_hdf(examples_h5, key=Utils.EXAMPLES, mode="r")

    return inputdata_folder, vocabulary_folder, definitions_df, examples_df, graph_folder


def graph_nodes(vocab_sources_ls=(F.WT2, F.SEMCOR), sp_method=Utils.SpMethod.FASTTEXT):
    Utils.init_logging("Test-GraphNodes.log")

    logging.info("Graph test. On: SenseLabeled corpus")
    graph_dataobj = DG.get_graph_dataobject(False, vocab_sources_ls, sp_method)

    inputdata_folder, vocabulary_folder, definitions_df, examples_df, graph_folder = \
        initialize_archives(vocab_sources_ls, sp_method)

    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, area_size=32, hops_in_area=1, graph_folder=graph_folder)

    last_idx_senses = graph_dataobj.node_types.tolist().index(1)
    last_idx_globals = graph_dataobj.node_types.tolist().index(2)
    last_idx_definitions = graph_dataobj.node_types.tolist().index(3)
    num_nodes = len(graph_dataobj.node_types.tolist())
    last_indices_tpl = (last_idx_senses , last_idx_globals, last_idx_definitions)
    logging.info("last_idx_senses=" + str(last_idx_senses))
    random_nodes = [6602 + last_idx_senses] # + \
                   #[random.randint(0, last_idx_senses) for _i in range(10)] + \
                   #[random.randint(last_idx_senses, last_idx_globals) for _i in range(10)] + \
                   #[random.randint(last_idx_globals, last_idx_definitions) for _i in range(5)] + \
                   #[random.randint(last_idx_definitions, num_nodes) for _i in range(5)]

    for center in random_nodes:
        logging.info("\nCenter:")
        log_node(center, last_indices_tpl, inputdata_folder, vocabulary_folder, definitions_df, examples_df)

        neighbours, _, _ = AD.get_node_data(grapharea_matrix, center, grapharea_size=32, features_mask=(True, False, False))
        logging.info("Neighbours:")
        for n_t in neighbours:
            log_node(n_t.item(), last_indices_tpl, inputdata_folder, vocabulary_folder, definitions_df, examples_df)


def get_missing_sense(vocab_sources_ls=(F.WT2, F.SEMCOR), sp_method=Utils.SpMethod.FASTTEXT):
    Utils.init_logging("Test-GraphNodes.log")

    graph_folder, inputdata_folder, vocabulary_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls,
                                                                                             sp_method)
    senseindices_db = sqlite3.connect(os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB))
    senseindices_db_c = senseindices_db.cursor()
    last_sense_idx = senseindices_db_c.execute("SELECT COUNT(*) from indices_table").fetchone()[0]
    #logging.info("last_sense_idx from db=" + str(last_sense_idx))
    first_idx_dummySenses = Utils.get_startpoint_dummySenses(inputdata_folder)

    graph_dataobj = DG.get_graph_dataobject(False, vocab_sources_ls, sp_method)
    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, area_size=32, hops_in_area=1, graph_folder=graph_folder)
    #logging.info("last_idx_senses from graph=" + str(graph_dataobj.node_types.tolist().index(1)))

    global_vocab_indices = [6602]
    for global_absolute_index in global_vocab_indices:
        logging.info(
        NI.get_missing_sense_label(global_absolute_index, grapharea_matrix, last_sense_idx, first_idx_dummySenses) )